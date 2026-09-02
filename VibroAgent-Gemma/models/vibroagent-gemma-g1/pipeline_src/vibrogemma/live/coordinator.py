from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any

from safetensors.torch import load_file

from ..config import resolve_from_config, resolve_static_asset
from ..exceptions import LiveAcquisitionError
from ..inference.engine import InferenceEngine
from ..inference.notifier import DesktopNotifier
from ..inference.policy import GeneratedOutputAlertPolicy
from ..inference.store import EventStore
from ..model.pipeline import VibroGemmaForClassification
from .ring_buffer import TimestampedRingBuffer
from .sources import LiveSource, ReplaySource, StdatalogUSBSource
from .sync import SynchronizedWindowAssembler


def load_runtime_model(config: dict[str, Any]) -> tuple[VibroGemmaForClassification, Path]:
    runtime_model = config["model"]
    generation = dict(config.get("generation") or {})
    for key in (
        "max_record_tokens",
        "max_explanation_tokens",
        "repetition_penalty",
        "confidence_calibration_path",
    ):
        if key in runtime_model:
            generation[key] = runtime_model[key]
    training = dict(config.get("training") or {})
    model_config = {
        "_config_path": config.get("_config_path", "configs/runtime.yaml"),
        "_config_dir": config.get("_config_dir", "configs"),
        "project": config.get("project", {}),
        "data": {
            "window_seconds": runtime_model.get("window_seconds", 10.0),
            "minimum_valid_fraction": 1.0 - float(config.get("live", {}).get("maximum_missing_fraction", 0.02)),
            "allow_local_training_data": False,
        },
        "model": {
            **runtime_model,
            "gemma_model_id": runtime_model["gemma_model_id"],
            "encoder_width": runtime_model.get("encoder_width", 192),
            "projector_hidden_multiplier": runtime_model.get("projector_hidden_multiplier", 2),
            "dropout": runtime_model.get("dropout", 0.05),
            "marker_text": runtime_model.get(
                "marker_text",
                {
                    "baseline_start": "\nREFERENCE BOARD:\n",
                    "baseline_end": "\nEND REFERENCE\n",
                    "target_start_template": "\nTARGET {index}:\n",
                    "target_end": "\nEND TARGET\n",
                },
            ),
            "system_prompt": runtime_model.get(
                "system_prompt",
                "Classify the six-board vibration episode from modal, wideband, physics and geometry tokens. Emit only supported labels.",
            ),
        },
        "labels": config.get("labels", {}),
        "training": {
            "language_loss": training.get("language_loss", {}),
            "structured_set_training": training.get("structured_set_training", {}),
        },
        "generation": generation,
        "final_protocol": config.get("final_protocol", {}),
    }
    model = VibroGemmaForClassification.from_config(model_config, training=False)
    checkpoint = resolve_from_config(config, runtime_model["checkpoint"])
    model.load_components(checkpoint, strict=False)
    adapter = checkpoint / "adapter"
    if adapter.exists():
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise LiveAcquisitionError("PEFT is required to load the runtime adapter") from exc
        model.gemma = PeftModel.from_pretrained(model.gemma, adapter)
        model.structured_generator.model = model.gemma
    elif (checkpoint / "gemma_trainable.safetensors").exists():
        model.gemma.load_state_dict(load_file(str(checkpoint / "gemma_trainable.safetensors")), strict=False)
    model.move_components_to_gemma_device()
    model.eval()
    return model, checkpoint


class LiveCoordinator:
    def __init__(
        self,
        *,
        source: LiveSource,
        buffers: dict[str, TimestampedRingBuffer],
        assembler: SynchronizedWindowAssembler,
        engine: InferenceEngine,
        stride_seconds: float,
    ) -> None:
        self.source = source
        self.buffers = buffers
        self.assembler = assembler
        self.engine = engine
        self.stride_seconds = float(stride_seconds)
        self.stop_event = Event()
        self.last_episode_id: str | None = None
        self.last_result: dict[str, Any] | None = None
        self.last_error: str | None = None

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        replay_path: str | Path | None = None,
        replay_real_time: bool = False,
    ) -> LiveCoordinator:
        model, checkpoint = load_runtime_model(config)
        live = config["live"]
        reference = str(live["reference_device_id"])
        targets = [str(value) for value in live["target_device_ids"]]
        physical_ids = [reference, *targets]
        raw_rate = float(live["original_sample_rate_hz"])
        buffers = {
            board_id: TimestampedRingBuffer(
                channels=3,
                sample_rate_hz=raw_rate,
                capacity_seconds=float(live.get("ring_buffer_seconds", 45.0)),
            )
            for board_id in physical_ids
        }
        assembler = SynchronizedWindowAssembler(
            buffers,
            reference_id=reference,
            target_ids=targets,
            raw_sample_rate_hz=raw_rate,
            window_seconds=float(config["model"]["window_seconds"]),
            maximum_missing_fraction=float(live.get("maximum_missing_fraction", 0.02)),
            max_clock_skew_ms=float(live.get("max_clock_skew_ms", 20.0)),
            clipping_abs_g=float(live.get("clipping_abs_g", 16.0)),
            sensor_useful_band_hz=float(live.get("sensor_useful_band_hz", 6000.0)),
            board_geometry=dict(live.get("board_geometry", {})),
            phase_synchronized=bool(live.get("synchronization", {}).get("phase_valid", False)),
            synchronization_verification_id=live.get("synchronization", {}).get("verification_id"),
        )
        if replay_path is not None:
            source: LiveSource = ReplaySource(
                replay_path,
                sample_rate_hz=raw_rate,
                real_time=replay_real_time,
            )
        else:
            source_name = str(live.get("source", "stdatalog"))
            if source_name != "stdatalog":
                raise LiveAcquisitionError(f"Unsupported live source: {source_name}")
            source = StdatalogUSBSource(live)
        service = config["service"]
        store = EventStore(resolve_from_config(config, service["database"]))
        policy_cfg = config.get("policy", {})
        policy = GeneratedOutputAlertPolicy(
            trigger_count=int(policy_cfg.get("trigger_count", 3)),
            trigger_window=int(policy_cfg.get("trigger_window", 5)),
            clear_after_consecutive_normal=int(policy_cfg.get("clear_after_consecutive_normal", 3)),
            cooldown_seconds=float(policy_cfg.get("popup_cooldown_seconds", 30.0)),
            alert_on_data_invalid=bool(policy_cfg.get("alert_on_data_invalid", True)),
        )
        schema_path = resolve_static_asset(config, "schemas/alert.schema.json")
        engine = InferenceEngine(
            model=model,
            checkpoint_name=str(checkpoint),
            store=store,
            policy=policy,
            notifier=DesktopNotifier(bool(service.get("desktop_notifications", True))),
            schema_path=schema_path,
            alert_window_dir=resolve_from_config(config, live.get("alert_directory", "events/windows")),
            persist_alert_windows=bool(live.get("persist_alert_windows", True)),
            persist_all_windows=bool(live.get("persist_all_windows", False)),
        )
        return cls(
            source=source,
            buffers=buffers,
            assembler=assembler,
            engine=engine,
            stride_seconds=float(config["model"].get("stride_seconds", 2.0)),
        )

    def ingest(
        self,
        board_id: str,
        values,
        start_time_s: float,
        timestamps_s=None,
        valid_mask=None,
        packet_loss_count: int = 0,
    ) -> None:
        if board_id not in self.buffers:
            raise LiveAcquisitionError(f"Received unconfigured board ID {board_id}")
        self.buffers[board_id].append(
            values,
            start_time_s=start_time_s,
            timestamps_s=timestamps_s,
            valid_mask=valid_mask,
            packet_loss_count=packet_loss_count,
        )

    def run_once(self) -> dict[str, Any] | None:
        if not self.assembler.ready():
            return None
        episode = self.assembler.assemble_latest()
        if episode.episode_id == self.last_episode_id:
            return None
        result, event_id, alert_emitted = self.engine.process(episode)
        self.last_episode_id = episode.episode_id
        self.last_result = {
            "event_id": event_id,
            "alert_emitted": alert_emitted,
            "result": result.to_dict(),
        }
        return self.last_result

    def run_forever(self) -> None:
        self.stop_event.clear()
        self.source.start(self.ingest)
        try:
            while not self.stop_event.wait(self.stride_seconds):
                try:
                    self.run_once()
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    raise
        finally:
            self.source.stop()


    def run_replay_until_complete(self) -> None:
        if not isinstance(self.source, ReplaySource):
            raise LiveAcquisitionError("run_replay_until_complete requires ReplaySource")
        self.stop_event.clear()
        self.source.start(self.ingest)
        try:
            while not self.source.finished and not self.stop_event.is_set():
                self.run_once()
                self.stop_event.wait(min(self.stride_seconds, 0.25))
            # Process the final aligned window after the last chunk arrives.
            self.run_once()
        finally:
            self.source.stop()

    def stop(self) -> None:
        self.stop_event.set()
        self.source.stop()
