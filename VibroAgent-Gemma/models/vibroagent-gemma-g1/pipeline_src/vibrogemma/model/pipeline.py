from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

from ..config import resolve_from_config
from ..contracts import ClassificationResult, SixBoardEpisode, TargetPrediction
from ..exceptions import ModelCompatibilityError
from ..signal.multiresolution import (
    MultiResolutionEpisodePreprocessor,
    PreparedEpisode,
    PreparedEpisodeBatch,
    prepared_episode_batch,
)
from ..utils import atomic_write_json, utc_now_iso
from .calibration import ConfidenceCalibrator
from .encoder import SixBoardVibrationTokenizer, TokenizerOutput
from .gemma_bridge import forward_with_mixed_embeddings, resolve_dtype, resolve_hidden_size
from .generation import GemmaStructuredGenerator
from .projector import VibrationToGemmaProjector
from .prompts import SoftTokenPromptBuilder
from .records import (
    ClassificationRecordCodec,
    GlobalPrediction,
    GlobalRecordCodec,
    TargetRecordCodec,
)
from .residual import CommonModeResidualAdapter
from .structured_set import (
    TARGET_COUNT,
    decision_statistics,
    decode_single_source_zone,
    decode_target_set,
    gather_binary_choice_logits,
    global_choice_loss,
    resolve_binary_choice_token_ids,
    single_source_zone_loss,
    structured_target_set_loss,
)


def _weighted_language_model_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute the weighted causal-LM loss without densifying ignored prompt logits.

    Only supervised token positions are gathered and promoted to float32. This is
    mathematically equivalent to the previous dense ``ignore_index`` formulation,
    while avoiding a batch × sequence × vocabulary float32 copy that can consume
    several GiB during residual-adapter training.
    """

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = loss_weights[:, 1:].contiguous()
    valid = shift_labels.ne(-100) & shift_weights.gt(0)
    if not torch.any(valid):
        raise RuntimeError("weighted language-model batch has no supervised tokens")

    supervised_logits = shift_logits[valid].float()
    supervised_labels = shift_labels[valid]
    supervised_weights = shift_weights[valid].float()
    token_loss = F.cross_entropy(
        supervised_logits,
        supervised_labels,
        reduction="none",
    )
    denominator = supervised_weights.sum().clamp_min(1e-8)
    return (token_loss * supervised_weights).sum() / denominator


def load_gemma_and_tokenizer(model_id: str, model_config: dict[str, Any], *, training: bool) -> tuple[Any, Any]:
    from transformers import AutoTokenizer

    dtype = resolve_dtype(str(model_config.get("dtype", "bfloat16")))
    revision = str(model_config.get("revision") or "").strip()
    tokenizer_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
    }
    if revision:
        tokenizer_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    common_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        "attn_implementation": model_config.get("attention_implementation", "sdpa"),
    }
    if revision:
        common_kwargs["revision"] = revision
    if not training:
        default_device = "auto" if torch.cuda.is_available() else "cpu"
        requested_device = str(model_config.get("device", default_device))
        if requested_device == "auto":
            common_kwargs["device_map"] = "auto"
        elif requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("model.device=cuda was requested but CUDA is unavailable")
            common_kwargs["device_map"] = {"": torch.cuda.current_device()}
        elif requested_device != "cpu":
            raise ValueError(f"Unsupported model.device value: {requested_device!r}")
    errors: list[str] = []
    loaders = []
    try:
        from transformers import AutoModelForCausalLM

        loaders.append(AutoModelForCausalLM)
    except ImportError:
        pass
    for class_name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        try:
            module = __import__("transformers", fromlist=[class_name])
            loaders.append(getattr(module, class_name))
        except (ImportError, AttributeError):
            continue
    for loader in loaders:
        try:
            model = loader.from_pretrained(model_id, **common_kwargs)
            return model, tokenizer
        except (ValueError, TypeError, OSError) as exc:
            errors.append(f"{loader.__name__}: {exc}")
    raise ModelCompatibilityError(
        "Unable to load Gemma with an available Transformers auto class. " + " | ".join(errors)
    )


def episode_tensors(
    episodes: Sequence[SixBoardEpisode | PreparedEpisode] | PreparedEpisodeBatch,
    device: torch.device,
    *,
    preprocessor: MultiResolutionEpisodePreprocessor | None = None,
) -> dict[str, torch.Tensor]:
    """Compatibility helper used by training code and external integrations."""

    if isinstance(episodes, PreparedEpisodeBatch):
        return episodes.to(device, non_blocking=True).tensors
    if preprocessor is None:
        if not all(isinstance(item, PreparedEpisode) for item in episodes):
            raise ValueError("A multi-resolution preprocessor is required for raw SixBoardEpisode inputs")
        first = episodes[0]
        assert isinstance(first, PreparedEpisode)
        # Any preprocessor would be unused when every item is already prepared.
        class _PreparedOnly:
            @staticmethod
            def prepare_many(items):
                return list(items)

        preprocessor = _PreparedOnly()  # type: ignore[assignment]
    return prepared_episode_batch(episodes, preprocessor=preprocessor, device=device).tensors


class VibroGemmaForClassification(nn.Module):
    """Three-branch vibration tokenizer + projector + Gemma generation.

    Gemma's language-model distribution is the only decision surface. The
    multi-resolution modules create continuous signal representations and
    auditable evidence; they do not emit vibration classes or severities.
    """

    def __init__(self, gemma: nn.Module, tokenizer: Any, config: dict[str, Any]) -> None:
        super().__init__()
        self.gemma = gemma
        self.hf_tokenizer = tokenizer
        self.config_payload = config
        model_config = config["model"]
        forbidden_config: list[str] = []
        configured_backend = model_config.get("decision_backend")
        if configured_backend is not None and str(configured_backend) != "gemma_lm":
            forbidden_config.append("model.decision_backend")
        if "direct_decision_head" in model_config:
            forbidden_config.append("model.direct_decision_head")
        if "direct_decision" in (config.get("training") or {}):
            forbidden_config.append("training.direct_decision")
        if "evidence_hybrid" in (config.get("generation") or {}):
            forbidden_config.append("generation.evidence_hybrid")
        if forbidden_config:
            raise ModelCompatibilityError(
                "Gemma must be the sole vibration classifier; alternate decision "
                "configuration is forbidden: " + ", ".join(forbidden_config)
            )
        self.preprocessor = MultiResolutionEpisodePreprocessor.from_config(config)
        multi = model_config.get("multiresolution", {})
        modal = multi.get("modal", {})
        wideband = multi.get("wideband", {})
        physics = multi.get("physics", {})
        fusion = multi.get("fusion", {})
        hidden_size = resolve_hidden_size(gemma)
        encoder_width = int(model_config.get("encoder_width", 192))
        self.vibration_tokenizer = SixBoardVibrationTokenizer(
            width=encoder_width,
            modal_depth=int(modal.get("depth", model_config.get("encoder_depth", 6))),
            modal_tokens_per_axis=int(modal.get("tokens_per_axis", 4)),
            modal_tokens_per_board=int(modal.get("tokens_per_board", 6)),
            modal_kernel_sizes=modal.get(
                "kernel_sizes", model_config.get("encoder_kernel_sizes", [7, 5, 5, 3, 3, 3])
            ),
            modal_dilations=modal.get(
                "dilations", model_config.get("encoder_dilations", [1, 2, 4, 8, 16, 32])
            ),
            wideband_input_channels=int(wideband.get("input_channels", 9)),
            wideband_depth=int(wideband.get("depth", 4)),
            wideband_tokens_per_board=int(wideband.get("tokens_per_board", 6)),
            wideband_frequency_patches=int(wideband.get("frequency_patches", 4)),
            wideband_time_patches=int(wideband.get("time_patches", 4)),
            physics_feature_count=len(self.preprocessor.physics_extractor.feature_names),
            physics_tokens_per_board=int(physics.get("tokens_per_board", 2)),
            attention_heads=int(fusion.get("attention_heads", 4)),
            branch_fusion_layers=int(fusion.get("branch_layers", 2)),
            cross_board_layers=int(fusion.get("cross_board_layers", 2)),
            dropout=float(model_config.get("dropout", 0.05)),
        )
        configured_token_count = model_config.get("tokens_per_board")
        if configured_token_count is not None and int(configured_token_count) != self.vibration_tokenizer.tokens_per_board:
            raise ValueError(
                "model.tokens_per_board must equal modal + wideband + physics branch tokens "
                f"({self.vibration_tokenizer.tokens_per_board})"
            )
        self.projector = VibrationToGemmaProjector(
            input_width=encoder_width,
            gemma_hidden_size=hidden_size,
            hidden_multiplier=int(model_config.get("projector_hidden_multiplier", 2)),
            dropout=float(model_config.get("dropout", 0.05)),
        )
        residual_cfg = dict(model_config.get("common_mode_residual") or {})
        self.common_mode_residual_adapter: CommonModeResidualAdapter | None = None
        if bool(residual_cfg.get("enabled", False)):
            self.common_mode_residual_adapter = CommonModeResidualAdapter(
                feature_count=len(self.preprocessor.physics_extractor.feature_names),
                width=encoder_width,
                physics_tokens_per_board=self.vibration_tokenizer.physics_tokens_per_board,
                attention_heads=int(residual_cfg.get("attention_heads", fusion.get("attention_heads", 4))),
                transformer_layers=int(residual_cfg.get("transformer_layers", 1)),
                dropout=float(residual_cfg.get("dropout", model_config.get("dropout", 0.05))),
                minimum_common_targets=int(residual_cfg.get("minimum_common_targets", 3)),
                maximum_gate=float(residual_cfg.get("maximum_gate", 0.5)),
                residual_clip=float(residual_cfg.get("residual_clip", 8.0)),
                scale_floor=float(residual_cfg.get("scale_floor", 1e-4)),
            )
        configured_classes = config.get("labels", {}).get("classes")
        configured_severities = config.get("labels", {}).get("severities")
        self.codec = (
            ClassificationRecordCodec(configured_classes, configured_severities or ("none", "advisory"))
            if configured_classes
            else ClassificationRecordCodec()
        )
        self.global_codec = GlobalRecordCodec(
            ("normal", "unknown_anomaly"),
            configured_severities or ("none", "advisory"),
        )
        self.target_codec = TargetRecordCodec(
            configured_classes or self.codec.classes,
            configured_severities or self.codec.severities,
        )
        self.structured_set_config = dict(
            config.get("training", {}).get("structured_set_training") or {}
        )
        structured_decision_separator = str(
            self.structured_set_config.get("decision_separator", " ")
        )
        language_loss = dict(config.get("training", {}).get("language_loss") or {})
        partial_label_training = dict(language_loss.get("partial_label_training") or {})
        decision_token_weighting = dict(
            language_loss.get("decision_token_weighting") or {}
        )
        self.prompt_builder = SoftTokenPromptBuilder(
            tokenizer,
            tokens_per_board=self.vibration_tokenizer.tokens_per_board,
            system_prompt=str(model_config["system_prompt"]),
            marker_text=dict(model_config["marker_text"]),
            codec=self.codec,
            global_codec=self.global_codec,
            target_codec=self.target_codec,
            classification_loss_weight=float(
                language_loss.get("classification_weight", 1.0)
            ),
            explanation_loss_weight=float(
                language_loss.get("explanation_weight", 1.0)
            ),
            eos_loss_weight=float(language_loss.get("eos_weight", 1.0)),
            global_task_weight=float(partial_label_training.get("global_task_weight", 1.0)),
            target_task_weight=float(partial_label_training.get("target_task_weight", 1.0)),
            joint_task_weight=float(partial_label_training.get("joint_task_weight", 1.0)),
            target_row_balance=bool(language_loss.get("target_row_balance", False)),
            prompt_prior_guard=bool(language_loss.get("prompt_prior_guard", False)),
            factorized_training=bool(partial_label_training.get("factorized", False)),
            target_decision_semantics=str(
                language_loss.get("target_decision_semantics", "response_anomaly")
            ),
            target_focus_layout=str(
                language_loss.get("target_focus_layout", "fixed")
            ),
            target_evidence_scope=str(
                language_loss.get("target_evidence_scope", "focus_only")
            ),
            classification_structure_weight=float(
                decision_token_weighting.get("structure_weight", 1.0)
            ),
            global_decision_weight=float(
                decision_token_weighting.get("global_decision_weight", 1.0)
            ),
            target_decision_weight=float(
                decision_token_weighting.get("target_decision_weight", 1.0)
            ),
            joint_decision_weight=float(
                decision_token_weighting.get("joint_decision_weight", 1.0)
            ),
            structured_decision_separator=structured_decision_separator,
            structured_target_query_order=str(
                self.structured_set_config.get("target_query_order", "fixed")
            ),
        )
        generation = config.get("generation", {})
        self.structured_set_training_enabled = bool(
            self.structured_set_config.get("enabled", False)
        )
        self.structured_set_decoding_enabled = bool(
            generation.get("structured_set_decoding", False)
        )
        self.single_source_zone_loss_weight = float(
            self.structured_set_config.get("single_source_zone_loss_weight", 0.0)
        )
        if self.single_source_zone_loss_weight < 0:
            raise ValueError("single_source_zone_loss_weight must be non-negative")
        self.legacy_set_objective_weight = float(
            self.structured_set_config.get("legacy_set_objective_weight", 1.0)
        )
        if self.legacy_set_objective_weight < 0:
            raise ValueError("legacy_set_objective_weight must be non-negative")
        self.single_source_zone_decoding_enabled = bool(
            generation.get("single_source_zone_decoding", False)
        )
        if (
            self.single_source_zone_decoding_enabled
            and not self.structured_set_decoding_enabled
        ):
            raise ValueError(
                "generation.single_source_zone_decoding requires "
                "generation.structured_set_decoding=true"
            )
        self.binary_choice_token_ids: tuple[int, int] | None = None
        if self.structured_set_training_enabled or self.structured_set_decoding_enabled:
            choices = tuple(
                str(value)
                for value in self.structured_set_config.get(
                    "choice_tokens", ("0", "1")
                )
            )
            self.binary_choice_token_ids = resolve_binary_choice_token_ids(
                tokenizer,
                choices,
                contexts=(
                    f"\nGLOBAL={structured_decision_separator}",
                    *(
                        f"\nT{index}={structured_decision_separator}"
                        for index in range(1, 6)
                    ),
                ),
            )
        self.structured_generator = GemmaStructuredGenerator(
            gemma,
            tokenizer,
            self.codec,
            global_codec=self.global_codec,
            target_codec=self.target_codec,
            max_record_tokens=int(generation.get("max_record_tokens", 160)),
            max_explanation_tokens=int(generation.get("max_explanation_tokens", 64)),
            repetition_penalty=float(generation.get("repetition_penalty", 1.05)),
        )
        self._last_common_mode_residual = None
        self.last_structured_loss_components: dict[str, torch.Tensor] | None = None
        self.last_decision_stats: dict[str, torch.Tensor] | None = None
        self.confidence_calibrator: ConfidenceCalibrator | None = None
        calibration_path = generation.get("confidence_calibration_path")
        if calibration_path:
            resolved = resolve_from_config(config, str(calibration_path))
            if resolved.exists():
                self.confidence_calibrator = ConfidenceCalibrator.load(resolved)

    @classmethod
    def from_config(cls, config: dict[str, Any], *, training: bool) -> VibroGemmaForClassification:
        model_id = str(config["model"]["gemma_model_id"])
        gemma, tokenizer = load_gemma_and_tokenizer(model_id, config["model"], training=training)
        instance = cls(gemma, tokenizer, config)
        if training and bool(config.get("training", {}).get("gradient_checkpointing", False)):
            if hasattr(instance.gemma, "gradient_checkpointing_enable"):
                instance.gemma.gradient_checkpointing_enable()
            if hasattr(instance.gemma.config, "use_cache"):
                instance.gemma.config.use_cache = False
        return instance

    @property
    def component_device(self) -> torch.device:
        try:
            return next(self.vibration_tokenizer.parameters()).device
        except StopIteration:
            return next(self.gemma.parameters()).device

    def move_components_to_gemma_device(self) -> None:
        from .gemma_bridge import resolve_text_model

        text_model = resolve_text_model(self.gemma)
        target = text_model.embed_tokens.weight.device
        self.vibration_tokenizer.to(target)
        self.projector.to(target)
        if self.common_mode_residual_adapter is not None:
            self.common_mode_residual_adapter.to(target)

    def prepare_batch(
        self,
        episodes: Sequence[SixBoardEpisode | PreparedEpisode] | PreparedEpisodeBatch,
    ) -> PreparedEpisodeBatch:
        return prepared_episode_batch(
            episodes,
            preprocessor=self.preprocessor,
            device=self.component_device,
        )

    def encode(
        self,
        episodes: Sequence[SixBoardEpisode | PreparedEpisode] | PreparedEpisodeBatch,
    ) -> tuple[torch.Tensor, TokenizerOutput, PreparedEpisodeBatch]:
        prepared_batch = self.prepare_batch(episodes)
        output = self.vibration_tokenizer(**prepared_batch.tensors)
        self._last_common_mode_residual = None
        encoded_tokens = output.tokens
        if self.common_mode_residual_adapter is not None:
            residual_correction = self.common_mode_residual_adapter(
                prepared_batch.tensors["physics_features"],
                prepared_batch.tensors["physics_mask"],
            )
            self._last_common_mode_residual = residual_correction
            board_tokens = output.board_tokens.clone()
            physics_start = self.vibration_tokenizer.tokens_per_board - self.vibration_tokenizer.physics_tokens_per_board
            board_tokens[:, :, physics_start:, :] = (
                board_tokens[:, :, physics_start:, :] + residual_correction
            )
            encoded_tokens = board_tokens.reshape(
                board_tokens.shape[0],
                board_tokens.shape[1] * board_tokens.shape[2],
                board_tokens.shape[3],
            )
            encoded_tokens = encoded_tokens * output.token_mask.unsqueeze(-1).to(encoded_tokens.dtype)
        soft_tokens = self.projector(encoded_tokens)
        soft_tokens = soft_tokens * output.token_mask.unsqueeze(-1).to(soft_tokens.dtype)
        return soft_tokens, output, prepared_batch

    def _structured_binary_forward(
        self,
        soft_tokens: torch.Tensor,
        prepared_batch: PreparedEpisodeBatch,
        *,
        use_cache: bool,
    ) -> tuple[Any, torch.Tensor]:
        if self.binary_choice_token_ids is None:
            raise RuntimeError("structured binary decision tokens were not initialized")
        structured = self.prompt_builder.build_structured_decision_batch(
            model=self.gemma,
            soft_tokens=soft_tokens,
            episodes=prepared_batch.episodes,
            evidence_texts=prepared_batch.evidence_texts,
        )
        output = forward_with_mixed_embeddings(
            self.gemma,
            structured.mixed_batch,
            use_cache=use_cache,
        )
        choice_logits = gather_binary_choice_logits(
            output.logits,
            structured.decision_positions,
            self.binary_choice_token_ids,
        )
        return output, choice_logits

    def _apply_structured_training_loss(
        self,
        output: Any,
        choice_logits: torch.Tensor,
        episodes: Sequence[SixBoardEpisode],
    ) -> Any:
        if choice_logits.shape != (len(episodes), 6, 2):
            raise RuntimeError(
                f"structured decision logits have unexpected shape {tuple(choice_logits.shape)}"
            )
        labels = []
        for episode in episodes:
            if episode.label is None:
                raise ValueError("structured-set training episode lacks a label")
            episode.label.validate()
            labels.append(episode.label)

        global_truth = torch.tensor(
            [label.global_class_name == "unknown_anomaly" for label in labels],
            device=choice_logits.device,
            dtype=torch.bool,
        )
        global_logits = choice_logits[:, 0, :]
        global_loss = global_choice_loss(global_logits, global_truth)

        supervised_indices = [
            index
            for index, label in enumerate(labels)
            if label.target_supervision_available
        ]
        target_logits: torch.Tensor | None = None
        target_truth: torch.Tensor | None = None
        zero = global_loss * 0.0
        set_loss = zero
        ranking_loss = zero
        positive_margin_loss = zero
        hard_negative_margin_loss = zero
        normal_margin_loss = zero
        class_balanced_binary_loss = zero
        legacy_set_objective_loss = zero
        single_source_zone_loss_value = zero
        target_predicted_mask: torch.Tensor | None = None
        structured_total = zero
        if supervised_indices:
            index_tensor = torch.tensor(
                supervised_indices,
                device=choice_logits.device,
                dtype=torch.long,
            )
            target_logits = choice_logits.index_select(0, index_tensor)[:, 1:, :]
            target_truth = torch.tensor(
                [
                    [target.affected for target in labels[index].targets]
                    for index in supervised_indices
                ],
                device=choice_logits.device,
                dtype=torch.bool,
            )
            set_result = structured_target_set_loss(
                target_logits,
                target_truth,
                false_positive_cost=float(
                    self.structured_set_config.get("false_positive_cost", 1.0)
                ),
                false_negative_cost=float(
                    self.structured_set_config.get("false_negative_cost", 1.0)
                ),
                cardinality_cost=float(
                    self.structured_set_config.get("cardinality_cost", 0.5)
                ),
                ranking_weight=float(
                    self.structured_set_config.get("ranking_loss_weight", 0.25)
                ),
                positive_margin_weight=float(
                    self.structured_set_config.get("positive_margin_loss_weight", 0.0)
                ),
                positive_margin=float(
                    self.structured_set_config.get("positive_target_margin", 1.0)
                ),
                hard_negative_margin_weight=float(
                    self.structured_set_config.get(
                        "hard_negative_margin_loss_weight", 0.0
                    )
                ),
                hard_negative_margin=float(
                    self.structured_set_config.get("hard_negative_margin", 1.0)
                ),
                normal_margin_weight=float(
                    self.structured_set_config.get("normal_margin_loss_weight", 0.0)
                ),
                normal_margin=float(
                    self.structured_set_config.get("normal_target_margin", 0.5)
                ),
                class_balanced_binary_weight=float(
                    self.structured_set_config.get(
                        "class_balanced_binary_loss_weight", 0.0
                    )
                ),
                margin_aggregation_temperature=float(
                    self.structured_set_config.get(
                        "margin_aggregation_temperature", 0.25
                    )
                ),
            )
            legacy_set_objective_loss = set_result.loss
            structured_total = (
                self.legacy_set_objective_weight * legacy_set_objective_loss
            )
            set_loss = set_result.set_loss
            ranking_loss = set_result.ranking_loss
            positive_margin_loss = set_result.positive_margin_loss
            hard_negative_margin_loss = set_result.hard_negative_margin_loss
            normal_margin_loss = set_result.normal_margin_loss
            class_balanced_binary_loss = set_result.class_balanced_binary_loss
            target_predicted_mask = set_result.predicted_mask
            if self.single_source_zone_loss_weight > 0:
                zone_result = single_source_zone_loss(target_logits, target_truth)
                single_source_zone_loss_value = zone_result.loss
                structured_total = (
                    structured_total
                    + self.single_source_zone_loss_weight
                    * single_source_zone_loss_value
                )
                if self.single_source_zone_decoding_enabled:
                    target_predicted_mask = zone_result.predicted_mask
            elif self.single_source_zone_decoding_enabled:
                target_predicted_mask, _, _ = decode_single_source_zone(target_logits)

        total_loss = (
            float(self.structured_set_config.get("global_loss_weight", 1.0))
            * global_loss
            + float(self.structured_set_config.get("set_loss_weight", 1.0))
            * structured_total
        )
        output.loss = total_loss
        output.binary_choice_logits = choice_logits
        structured_loss_components = {
            "global": global_loss.detach(),
            "set": set_loss.detach(),
            "ranking": ranking_loss.detach(),
            "positive_margin": positive_margin_loss.detach(),
            "hard_negative_margin": hard_negative_margin_loss.detach(),
            "normal_margin": normal_margin_loss.detach(),
            "class_balanced_binary": class_balanced_binary_loss.detach(),
            "legacy_set_objective": legacy_set_objective_loss.detach(),
            "total": total_loss.detach(),
        }
        if self.single_source_zone_loss_weight > 0:
            structured_loss_components["single_source_zone"] = (
                single_source_zone_loss_value.detach()
            )
        decision_stats = decision_statistics(
            global_logits.detach(),
            global_truth,
            target_logits.detach() if target_logits is not None else None,
            target_truth,
            target_predicted_mask=(
                target_predicted_mask.detach()
                if target_predicted_mask is not None
                else None
            ),
        )
        # Accelerate's mixed-precision output conversion reconstructs the declared
        # Transformers ModelOutput fields and drops dynamically attached attributes.
        # Keep a detached model-side copy so logging and collapse guards remain live.
        self.last_structured_loss_components = structured_loss_components
        self.last_decision_stats = decision_stats
        output.structured_loss_components = structured_loss_components
        output.decision_stats = decision_stats
        return output

    def forward(
        self,
        episodes: Sequence[SixBoardEpisode | PreparedEpisode] | PreparedEpisodeBatch,
    ) -> Any:
        soft_tokens, _, prepared_batch = self.encode(episodes)
        if self.structured_set_training_enabled:
            output, choice_logits = self._structured_binary_forward(
                soft_tokens,
                prepared_batch,
                use_cache=False,
            )
            return self._apply_structured_training_loss(
                output,
                choice_logits,
                prepared_batch.episodes,
            )
        batch = self.prompt_builder.build_batch(
            model=self.gemma,
            soft_tokens=soft_tokens,
            episodes=prepared_batch.episodes,
            evidence_texts=prepared_batch.evidence_texts,
            evidences=prepared_batch.evidences,
            training=True,
        )
        output = forward_with_mixed_embeddings(self.gemma, batch, use_cache=False)
        if batch.labels is not None and batch.loss_weights is not None:
            output.loss = _weighted_language_model_loss(
                output.logits,
                batch.labels,
                batch.loss_weights,
            )
        return output

    @torch.inference_mode()
    def classify(
        self,
        episode: SixBoardEpisode | PreparedEpisode,
        *,
        checkpoint_name: str = "unknown",
        generate_explanation: bool = True,
    ) -> ClassificationResult:
        self.eval()
        soft_tokens, _, prepared_batch = self.encode([episode])
        prepared = prepared_batch.require_full_prepared()[0]
        generation_cfg = dict(self.config_payload.get("generation") or {})
        factorized_decoding = bool(generation_cfg.get("factorized_decoding", False))
        structured_set_decoding = bool(
            generation_cfg.get("structured_set_decoding", False)
        )
        single_source_zone_decoding = bool(
            generation_cfg.get("single_source_zone_decoding", False)
        )
        global_prediction = None
        global_record = None
        global_probability = None
        raw_global_probability = None
        global_confidence_operational = False
        target_records: list[str] = []
        raw_explanation = ""
        structured_candidate_scores: list[float] | None = None
        single_source_zone_candidate_scores: list[float] | None = None
        gemma_binary_choice_slot_order: list[str] | None = None
        gemma_binary_choice_token_ids: list[int] | None = None
        gemma_binary_choice_logits: list[list[float]] | None = None
        if structured_set_decoding:
            _output, choice_logits = self._structured_binary_forward(
                soft_tokens,
                prepared_batch,
                use_cache=False,
            )
            decisions = choice_logits[0]
            if self.binary_choice_token_ids is None:
                raise RuntimeError(
                    "structured Gemma choice-token IDs were not initialized"
                )
            gemma_binary_choice_slot_order = [
                "GLOBAL",
                *(f"T{index}" for index in range(1, TARGET_COUNT + 1)),
            ]
            gemma_binary_choice_token_ids = [
                int(value) for value in self.binary_choice_token_ids
            ]
            gemma_binary_choice_logits = [
                [float(value) for value in pair]
                for pair in decisions.detach().cpu().tolist()
            ]
            global_probabilities = torch.softmax(decisions[0], dim=-1)
            global_is_anomaly = bool(int(torch.argmax(decisions[0]).item()))
            raw_global_probability = float(
                global_probabilities[1 if global_is_anomaly else 0].item()
            )
            global_prediction = GlobalPrediction(
                "unknown_anomaly" if global_is_anomaly else "normal",
                "advisory" if global_is_anomaly else "none",
                raw_global_probability,
            )
            global_record = (
                f"{self.global_codec.header}\n"
                f"G|{global_prediction.class_name}|{global_prediction.severity}\nEND"
            )
            if (
                self.confidence_calibrator is not None
                and self.confidence_calibrator.global_operational
            ):
                global_probability = self.confidence_calibrator.calibrate_global(
                    raw_global_probability
                )
                global_confidence_operational = global_probability is not None
            else:
                global_probability = None
            cardinality_bias = generation_cfg.get("target_set_cardinality_bias")
            if single_source_zone_decoding:
                decoded_mask, affected_probabilities, candidate_scores = (
                    decode_single_source_zone(
                        decisions[1:, :].unsqueeze(0),
                        cardinality_bias=cardinality_bias,
                    )
                )
                single_source_zone_candidate_scores = [
                    float(value)
                    for value in candidate_scores[0].detach().cpu().tolist()
                ]
            else:
                decoded_mask, affected_probabilities, candidate_scores = decode_target_set(
                    decisions[1:, :].unsqueeze(0),
                    cardinality_bias=cardinality_bias,
                )
                structured_candidate_scores = [
                    float(value)
                    for value in candidate_scores[0].detach().cpu().tolist()
                ]
            predictions = []
            for target_index in range(1, TARGET_COUNT + 1):
                affected = bool(decoded_mask[0, target_index - 1].item())
                affected_probability = float(
                    affected_probabilities[0, target_index - 1].item()
                )
                selected_probability = (
                    affected_probability if affected else 1.0 - affected_probability
                )
                prediction = TargetPrediction(
                    sensor_id=f"target_{target_index}",
                    affected=affected,
                    class_name="unknown_anomaly" if affected else "normal",
                    severity="advisory" if affected else "none",
                    choice_probability=selected_probability,
                    raw_choice_probability=selected_probability,
                    confidence_operational=False,
                )
                prediction.validate()
                predictions.append(prediction)
                target_records.append(
                    f"{self.target_codec.header}\n"
                    f"T{target_index}|{1 if affected else 0}|"
                    f"{prediction.class_name}|{prediction.severity}\nEND"
                )
            assembled_record = self.codec.assemble(predictions)
        elif factorized_decoding:
            global_batch = self.prompt_builder.build_batch(
                model=self.gemma,
                soft_tokens=soft_tokens,
                episodes=[prepared.episode],
                evidence_texts=[prepared.evidence_text],
                evidences=[prepared.evidence],
                training=False,
                inference_task="global",
            )
            global_prediction, global_output = self.structured_generator.generate_global(
                global_batch
            )
            global_record = global_output.record_text
            raw_global_probability = global_output.choice_probability
            if (
                self.confidence_calibrator is not None
                and self.confidence_calibrator.global_operational
            ):
                global_probability = self.confidence_calibrator.calibrate_global(
                    raw_global_probability
                )
                global_confidence_operational = global_probability is not None
            else:
                global_probability = None
            predictions = []
            for target_index in range(1, 6):
                target_batch = self.prompt_builder.build_batch(
                    model=self.gemma,
                    soft_tokens=soft_tokens,
                    episodes=[prepared.episode],
                    evidence_texts=[prepared.evidence_text],
                    evidences=[prepared.evidence],
                    training=False,
                    inference_task="target",
                    focus_target_index=target_index,
                )
                target_prediction, target_output = (
                    self.structured_generator.generate_target(
                        target_batch, target_index=target_index
                    )
                )
                predictions.append(target_prediction)
                target_records.append(target_output.record_text)
            assembled_record = self.codec.assemble(predictions)
        else:
            batch = self.prompt_builder.build_batch(
                model=self.gemma,
                soft_tokens=soft_tokens,
                episodes=[prepared.episode],
                evidence_texts=[prepared.evidence_text],
                evidences=[prepared.evidence],
                training=False,
            )
            generated = self.structured_generator.generate(
                batch,
                allowed_numeric_text=prepared.evidence_text,
                generate_explanation=generate_explanation,
            )
            predictions = self.codec.parse(
                generated.record_text, generated.target_probabilities
            )
            assembled_record = generated.record_text
            raw_explanation = generated.explanation
            inferred_anomaly = any(
                prediction.class_name not in {"normal", "data_invalid"}
                for prediction in predictions
            )
            global_prediction = GlobalPrediction(
                "unknown_anomaly" if inferred_anomaly else "normal",
                self.codec.system_state(predictions) if inferred_anomaly else "none",
                None,
            )

        # Deterministic quality gates are not an anomaly classifier. They only
        # prevent Gemma from diagnosing unusable reference/target data.
        invalid_targets: list[str] = []
        reference_invalid = not prepared.episode.boards[0].quality.is_valid
        for target_index, board in enumerate(prepared.episode.boards[1:], start=1):
            if reference_invalid or not board.quality.is_valid:
                invalid_targets.append(f"target_{target_index}")
                predictions[target_index - 1] = TargetPrediction(
                    sensor_id=f"target_{target_index}",
                    affected=False,
                    class_name="data_invalid",
                    severity="advisory",
                    choice_probability=None,
                    raw_choice_probability=None,
                    confidence_operational=False,
                )

        # Raw branch-choice probabilities are always retained for audit. They
        # are exposed as operational confidence only when a validation-fitted
        # calibrator passed its discrimination gate.
        for prediction in predictions:
            raw = prediction.raw_choice_probability
            if raw is None:
                raw = prediction.choice_probability
            prediction.raw_choice_probability = raw
            if self.confidence_calibrator is not None and self.confidence_calibrator.operational:
                prediction.choice_probability = self.confidence_calibrator.calibrate(raw)
                prediction.confidence_operational = prediction.choice_probability is not None
            else:
                prediction.choice_probability = None
                prediction.confidence_operational = False

        quality = {
            board.board_id: {
                "valid": board.quality.is_valid,
                "valid_fraction": board.quality.valid_fraction,
                "missing_fraction": board.quality.missing_fraction,
                "clipped_fraction": board.quality.clipped_fraction,
                "axes_present": list(board.quality.axes_present),
                "usable_band_hz": board.quality.usable_band_hz,
                "packet_loss_count": board.quality.packet_loss_count,
                "clock_skew_ms": board.quality.clock_skew_ms,
                "orientation_calibrated": board.metadata.get("orientation_calibrated", False),
                "orientation_applied": board.metadata.get("orientation_applied", False),
                "notes": list(board.quality.notes),
            }
            for board in prepared.episode.boards
        }
        explanation_decision = self.prompt_builder.explanation_renderer.choose(
            raw_explanation,
            predictions,
            prepared.evidence,
            invalid_targets=invalid_targets,
        )
        effective_duration = float(
            prepared.episode.boards[0].values.shape[1]
            / prepared.episode.boards[0].sample_rate_hz
        )
        nominal_duration = float(self.config_payload["data"]["window_seconds"])
        assert global_prediction is not None
        predicted_affected = [
            prediction.sensor_id for prediction in predictions if prediction.affected
        ]
        global_target_consistent = bool(
            (global_prediction.class_name == "normal" and not predicted_affected)
            or (
                global_prediction.class_name == "unknown_anomaly"
                and bool(predicted_affected)
            )
        )
        if global_prediction.class_name == "normal" and not predicted_affected:
            localization_status = "normal"
        elif not predicted_affected:
            localization_status = "unresolved_global_anomaly"
        elif len(predicted_affected) == 5:
            localization_status = "all_five_targets"
        else:
            localization_status = "partial_target_set"
        result = ClassificationResult(
            schema_version="1.0",
            window={
                "start_utc": str(prepared.episode.metadata.get("start_utc", utc_now_iso())),
                "duration_seconds": nominal_duration,
                "effective_duration_seconds": effective_duration,
                "episode_id": prepared.episode.episode_id,
                "start_time_s": float(prepared.episode.boards[0].start_time_s),
            },
            system_state=self.codec.system_state(
                predictions,
                global_class_name=global_prediction.class_name,
                global_severity=global_prediction.severity,
            ),
            targets=predictions,
            explanation=explanation_decision.text,
            quality=quality,
            model={
                "checkpoint": checkpoint_name,
                "base_model": str(self.config_payload["model"]["gemma_model_id"]),
                "raw_record": assembled_record,
                "raw_global_record": global_record,
                "raw_target_records": target_records,
                "raw_explanation": raw_explanation,
                "decision_mode": (
                    (
                        "parallel_binary_global_plus_single_source_zone_v1"
                        if single_source_zone_decoding
                        else "parallel_binary_global_plus_joint_target_set_v1"
                    )
                    if structured_set_decoding
                    else (
                        "factorized_global_plus_independent_targets_v1"
                        if factorized_decoding
                        else "joint_autoregressive_rows_legacy"
                    )
                ),
                "global_class": global_prediction.class_name,
                "global_severity": global_prediction.severity,
                "raw_global_choice_probability": raw_global_probability,
                "global_choice_probability": global_probability,
                "global_confidence_operational": global_confidence_operational,
                "global_target_consistent": global_target_consistent,
                "localization_status": localization_status,
                "structured_target_set_candidate_scores": structured_candidate_scores,
                "single_source_zone_candidate_scores": (
                    single_source_zone_candidate_scores
                ),
                "gemma_binary_choice_slot_order": gemma_binary_choice_slot_order,
                "gemma_binary_choice_token_ids": gemma_binary_choice_token_ids,
                "gemma_binary_choice_logits": gemma_binary_choice_logits,
                "explanation_source": explanation_decision.source,
                "explanation_grounded": explanation_decision.grounded,
                "confidence_calibration": (
                    {
                        "path": self.confidence_calibrator.source_path,
                        "target_operational": self.confidence_calibrator.operational,
                        "target_validation_auroc": self.confidence_calibrator.validation_auroc,
                        "global_operational": self.confidence_calibrator.global_operational,
                        "global_validation_auroc": self.confidence_calibrator.global_validation_auroc,
                    }
                    if self.confidence_calibrator is not None
                    else {
                        "path": None,
                        "target_operational": False,
                        "target_validation_auroc": None,
                        "global_operational": False,
                        "global_validation_auroc": None,
                    }
                ),
                "tokens_per_board": self.vibration_tokenizer.tokens_per_board,
                "modal_tokens_per_board": self.vibration_tokenizer.modal_tokens_per_board,
                "wideband_tokens_per_board": self.vibration_tokenizer.wideband_tokens_per_board,
                "physics_tokens_per_board": self.vibration_tokenizer.physics_tokens_per_board,
                "common_mode_residual": (
                    {
                        "enabled": True,
                        "gate_values": [
                            float(value)
                            for value in self.common_mode_residual_adapter.gate_values.detach().cpu().tolist()
                        ],
                        "maximum_absolute_gate": float(
                            self.common_mode_residual_adapter.gate_values.detach().float().abs().max().item()
                        ),
                        "correction_rms": (
                            float(
                                self._last_common_mode_residual.detach()
                                .float()
                                .pow(2)
                                .mean()
                                .sqrt()
                                .item()
                            )
                            if self._last_common_mode_residual is not None
                            else 0.0
                        ),
                        "correction_max_abs": (
                            float(
                                self._last_common_mode_residual.detach()
                                .float()
                                .abs()
                                .max()
                                .item()
                            )
                            if self._last_common_mode_residual is not None
                            else 0.0
                        ),
                    }
                    if self.common_mode_residual_adapter is not None
                    else {
                        "enabled": False,
                        "gate_values": [],
                        "maximum_absolute_gate": 0.0,
                        "correction_rms": 0.0,
                        "correction_max_abs": 0.0,
                    }
                ),
            },
            evidence=prepared.evidence,
        )
        result.validate()
        return result


    def common_mode_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return correction-energy and gate-magnitude penalties for the last forward pass."""

        if self.common_mode_residual_adapter is None:
            zero = torch.zeros((), device=self.component_device)
            return zero, zero
        output = self._last_common_mode_residual
        if output is None:
            zero = torch.zeros((), device=self.component_device)
            return zero, self.common_mode_residual_adapter.gate_values.abs().mean()
        correction_energy = output.float().pow(2).mean()
        gate_magnitude = self.common_mode_residual_adapter.gate_values.float().abs().mean()
        return correction_energy, gate_magnitude

    def freeze_for_common_mode_adapter(self) -> None:
        """Freeze the champion and expose only the residual adapter parameters."""

        for parameter in self.parameters():
            parameter.requires_grad = False
        if self.common_mode_residual_adapter is None:
            raise RuntimeError("model.common_mode_residual.enabled must be true")
        for parameter in self.common_mode_residual_adapter.parameters():
            parameter.requires_grad = True

    def freeze_gemma(self) -> None:
        for parameter in self.gemma.parameters():
            parameter.requires_grad = False

    def unfreeze_gemma(self) -> None:
        for parameter in self.gemma.parameters():
            parameter.requires_grad = True

    def component_state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for name, tensor in self.vibration_tokenizer.state_dict().items():
            state[f"vibration_tokenizer.{name}"] = tensor.detach().cpu().contiguous()
        for name, tensor in self.projector.state_dict().items():
            state[f"projector.{name}"] = tensor.detach().cpu().contiguous()
        if self.common_mode_residual_adapter is not None:
            for name, tensor in self.common_mode_residual_adapter.state_dict().items():
                state[f"common_mode_residual_adapter.{name}"] = tensor.detach().cpu().contiguous()
        return state

    def save_components(self, directory: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        save_file(self.component_state_dict(), str(destination / "vibration_components.safetensors"))
        component_config: dict[str, Any] = {
            "version": str(
                self.config_payload.get("project", {}).get("version", "unknown")
            ),
            "model": self.config_payload["model"],
            "labels": self.config_payload.get("labels", {}),
            "generation": self.config_payload.get("generation", {}),
            "physics_feature_names": list(
                self.preprocessor.physics_extractor.feature_names
            ),
            "metadata": metadata or {},
        }
        atomic_write_json(
            destination / "vibrogemma_config.json",
            component_config,
        )
        if hasattr(self.hf_tokenizer, "save_pretrained"):
            self.hf_tokenizer.save_pretrained(destination / "tokenizer")

    def load_components(self, directory: str | Path, *, strict: bool = True) -> None:
        from ..training.checkpoint import validate_checkpoint_data_contract

        validate_checkpoint_data_contract(self.config_payload, directory)
        source = Path(directory) / "vibration_components.safetensors"
        state = load_file(str(source))
        tokenizer_state = {
            key.removeprefix("vibration_tokenizer."): value
            for key, value in state.items()
            if key.startswith("vibration_tokenizer.")
        }
        projector_state = {
            key.removeprefix("projector."): value
            for key, value in state.items()
            if key.startswith("projector.")
        }
        residual_state = {
            key.removeprefix("common_mode_residual_adapter."): value
            for key, value in state.items()
            if key.startswith("common_mode_residual_adapter.")
        }
        self.vibration_tokenizer.load_state_dict(tokenizer_state, strict=strict)
        self.projector.load_state_dict(projector_state, strict=strict)
        if self.common_mode_residual_adapter is not None and residual_state:
            self.common_mode_residual_adapter.load_state_dict(residual_state, strict=strict)
        elif strict and self.common_mode_residual_adapter is not None and not residual_state:
            raise RuntimeError("checkpoint is missing common-mode residual adapter weights")
