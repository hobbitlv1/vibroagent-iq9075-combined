from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from ..config import resolve_from_config, resolve_static_asset
from ..model.pipeline import VibroGemmaForClassification
from ..utils import atomic_write_json, sha256_file, utc_now_iso


class EncoderProjectorExport(torch.nn.Module):
    def __init__(self, model: VibroGemmaForClassification) -> None:
        super().__init__()
        self.tokenizer = model.vibration_tokenizer
        self.projector = model.projector
        self.common_mode_residual_adapter = model.common_mode_residual_adapter

    def forward(
        self,
        modal_values: torch.Tensor,
        modal_axis_mask: torch.Tensor,
        modal_sample_mask: torch.Tensor,
        wideband_features: torch.Tensor,
        wideband_frequency_mask: torch.Tensor,
        wideband_cross_valid: torch.Tensor,
        physics_features: torch.Tensor,
        physics_mask: torch.Tensor,
        sensor_positions: torch.Tensor,
        orientation_features: torch.Tensor,
        quality_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.tokenizer(
            modal_values=modal_values,
            modal_axis_mask=modal_axis_mask,
            modal_sample_mask=modal_sample_mask,
            wideband_features=wideband_features,
            wideband_frequency_mask=wideband_frequency_mask,
            wideband_cross_valid=wideband_cross_valid,
            physics_features=physics_features,
            physics_mask=physics_mask,
            sensor_positions=sensor_positions,
            orientation_features=orientation_features,
            quality_features=quality_features,
        )
        encoded_tokens = output.tokens
        if self.common_mode_residual_adapter is not None:
            residual_correction = self.common_mode_residual_adapter(
                physics_features,
                physics_mask,
            )
            board_tokens = output.board_tokens.clone()
            physics_start = self.tokenizer.tokens_per_board - self.tokenizer.physics_tokens_per_board
            board_tokens[:, :, physics_start:, :] = (
                board_tokens[:, :, physics_start:, :] + residual_correction
            )
            encoded_tokens = board_tokens.reshape(
                board_tokens.shape[0],
                board_tokens.shape[1] * board_tokens.shape[2],
                board_tokens.shape[3],
            )
            encoded_tokens = encoded_tokens * output.token_mask.unsqueeze(-1).to(encoded_tokens.dtype)
        projected = self.projector(encoded_tokens)
        return projected, output.token_mask


def _load_checkpoint(model: VibroGemmaForClassification, checkpoint: Path) -> None:
    model.load_components(checkpoint, strict=False)
    adapter = checkpoint / "adapter"
    if adapter.exists():
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required to load the checkpoint adapter") from exc
        model.gemma = PeftModel.from_pretrained(model.gemma, adapter)
        model.structured_generator.model = model.gemma
    elif (checkpoint / "gemma_trainable.safetensors").exists():
        model.gemma.load_state_dict(load_file(str(checkpoint / "gemma_trainable.safetensors")), strict=False)


def export_deployment_bundle(
    config: dict[str, Any],
    *,
    checkpoint: str | Path,
    output_dir: str | Path | None = None,
    merge_lora: bool = True,
    export_onnx: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else resolve_from_config(config, "models/vibrogemma-final")
    )
    destination.mkdir(parents=True, exist_ok=True)
    model = VibroGemmaForClassification.from_config(config, training=False)
    _load_checkpoint(model, checkpoint_path)
    model.move_components_to_gemma_device()
    model.eval()
    model.save_components(destination, metadata={"source_checkpoint": str(checkpoint_path)})

    gemma_dir = destination / "gemma"
    gemma = model.gemma
    if merge_lora and hasattr(gemma, "merge_and_unload"):
        gemma = gemma.merge_and_unload(safe_merge=True)
    if hasattr(gemma, "save_pretrained"):
        gemma.save_pretrained(gemma_dir, safe_serialization=True)
    model.hf_tokenizer.save_pretrained(destination / "tokenizer")

    onnx_path = destination / "vibration_encoder_projector.onnx"
    if export_onnx:
        settings = model.preprocessor.settings
        modal_samples = int(round(settings.modal_sample_rate_hz * settings.window_seconds))
        frequency_bins = settings.wideband_frequency_bins
        time_bins = settings.wideband_time_bins
        feature_count = len(model.preprocessor.physics_extractor.feature_names)
        wrapper = EncoderProjectorExport(model).to("cpu").eval()
        dummy = (
            torch.zeros(1, 6, 3, modal_samples, dtype=torch.float32),
            torch.ones(1, 6, 3, dtype=torch.bool),
            torch.ones(1, 6, modal_samples, dtype=torch.bool),
            torch.zeros(1, 6, 3, 9, frequency_bins, time_bins, dtype=torch.float32),
            torch.ones(1, 6, frequency_bins, dtype=torch.bool),
            torch.zeros(1, 6, 3, dtype=torch.bool),
            torch.zeros(1, 6, feature_count, dtype=torch.float32),
            torch.ones(1, 6, feature_count, dtype=torch.bool),
            torch.full((1, 6, 3), float("nan"), dtype=torch.float32),
            torch.zeros(1, 6, 11, dtype=torch.float32),
            torch.zeros(1, 6, 13, dtype=torch.float32),
        )
        input_names = [
            "modal_values",
            "modal_axis_mask",
            "modal_sample_mask",
            "wideband_features",
            "wideband_frequency_mask",
            "wideband_cross_valid",
            "physics_features",
            "physics_mask",
            "sensor_positions",
            "orientation_features",
            "quality_features",
        ]
        # The live edge path processes one complete six-board episode at a
        # time. A static batch-one graph is more robust for ONNX and for later
        # QNN/QAIRT compilation than a symbolic dynamic batch.
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_path,
            input_names=input_names,
            output_names=["soft_tokens", "soft_token_mask"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=True,
        )

        import onnx

        onnx_model = onnx.load(str(onnx_path), load_external_data=True)
        onnx.checker.check_model(onnx_model)

        import numpy as np
        import onnxruntime as ort

        with torch.inference_mode():
            expected_tokens, expected_mask = wrapper(*dummy)
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        ort_inputs = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(input_names, dummy, strict=True)
        }
        actual_tokens, actual_mask = session.run(None, ort_inputs)
        expected_tokens_np = expected_tokens.detach().cpu().numpy()
        expected_mask_np = expected_mask.detach().cpu().numpy()
        np.testing.assert_allclose(
            actual_tokens,
            expected_tokens_np,
            rtol=2e-3,
            atol=2e-4,
        )
        if not np.array_equal(actual_mask, expected_mask_np):
            raise RuntimeError("ONNX soft-token mask differs from the PyTorch export wrapper")
        atomic_write_json(
            destination / "onnx_validation.json",
            {
                "checker_passed": True,
                "runtime_provider": "CPUExecutionProvider",
                "tokens_shape": list(actual_tokens.shape),
                "mask_shape": list(actual_mask.shape),
                "max_absolute_token_error": float(
                    np.max(np.abs(actual_tokens - expected_tokens_np))
                ),
                "mean_absolute_token_error": float(
                    np.mean(np.abs(actual_tokens - expected_tokens_np))
                ),
                "rtol": 2e-3,
                "atol": 2e-4,
            },
        )

    schema_source = resolve_static_asset(config, "schemas/alert.schema.json")
    shutil.copy2(schema_source, destination / "alert.schema.json")
    calibration_source = config.get("generation", {}).get("confidence_calibration_path")
    calibration_copied = False
    if calibration_source:
        calibration_path = resolve_from_config(config, str(calibration_source))
        if calibration_path.exists():
            shutil.copy2(calibration_path, destination / "confidence_calibrator.json")
            calibration_copied = True

    # Make the component configuration portable. Training configuration paths
    # may point into a Colab/Drive workspace, but a deployment bundle must not
    # require that original absolute path. Relative paths resolve from the
    # exported bundle directory when the component config is loaded later.
    component_config_path = destination / "vibrogemma_config.json"
    if component_config_path.exists():
        component_config = json.loads(component_config_path.read_text(encoding="utf-8"))
        generation_config = component_config.setdefault("generation", {})
        generation_config["confidence_calibration_path"] = (
            "confidence_calibrator.json" if calibration_copied else None
        )
        atomic_write_json(component_config_path, component_config)

    settings = model.preprocessor.settings
    atomic_write_json(
        destination / "token_layout.json",
        {
            "boards": 6,
            "reference_index": 0,
            "target_indices": [1, 2, 3, 4, 5],
            "tokens_per_board": model.vibration_tokenizer.tokens_per_board,
            "modal_tokens_per_board": model.vibration_tokenizer.modal_tokens_per_board,
            "wideband_tokens_per_board": model.vibration_tokenizer.wideband_tokens_per_board,
            "physics_tokens_per_board": model.vibration_tokenizer.physics_tokens_per_board,
            "total_soft_tokens": model.vibration_tokenizer.tokens_per_board * 6,
            "modal_sample_rate_hz": settings.modal_sample_rate_hz,
            "wideband_max_frequency_hz": settings.wideband_max_frequency_hz,
            "wideband_frequency_bins": settings.wideband_frequency_bins,
            "wideband_time_bins": settings.wideband_time_bins,
            "window_seconds": settings.window_seconds,
            "axes": ["x", "y", "z"],
            "axis_handling": "genuine axes with masks and learned axis embeddings",
            "orientation": "sensor-to-building rotation when full XYZ is present; calibrated matrix always embedded",
            "cross_spectral_gate": "cross-power, phase and coherence only when synchronization is explicitly verified",
            "physics_feature_names": list(model.preprocessor.physics_extractor.feature_names),
            "common_mode_residual": (
                {
                    "enabled": True,
                    "gate_values": [
                        float(value)
                        for value in model.common_mode_residual_adapter.gate_values.detach().cpu().tolist()
                    ],
                    "correction_slots": "existing physics-token slots",
                }
                if model.common_mode_residual_adapter is not None
                else {"enabled": False, "gate_values": []}
            ),
        },
    )
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    manifest = {
        "bundle_version": "2.2",
        "created_utc": utc_now_iso(),
        "source_checkpoint": str(checkpoint_path),
        "base_model": config["model"]["gemma_model_id"],
        "files": [
            {
                "path": str(path.relative_to(destination)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
        "deployment_profile": "binary_state_v2",
        "supported_classes": list(config.get("labels", {}).get("classes", [])),
        "supported_severities": list(config.get("labels", {}).get("severities", [])),
        "onnx_batch_size": 1,
        "geniex_ready": False,
        "geniex_note": "This bundle targets native Transformers. GenieX requires the external-embedding API patch.",
    }
    atomic_write_json(destination / "bundle_manifest.json", manifest)
    return manifest
