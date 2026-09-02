from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from ..config import resolve_from_config
from ..exceptions import DataContractError
from ..signal.multiresolution import MultiResolutionEpisodePreprocessor, MultiResolutionSettings
from ..signal.profile import HealthyProfileBuilder
from ..utils import atomic_write_json, utc_now_iso
from .episodes import EpisodeDataset


def fit_healthy_profile(config: dict[str, Any], *, split: str = "train") -> dict[str, Any]:
    """Fit target-location healthy statistics from public, labelled, group-isolated episodes.

    This produces numeric reference statistics only. It never chooses an anomaly
    class, threshold, severity, or alert.
    """

    index_path = resolve_from_config(config, config["data"]["episode_root"]) / "index.jsonl"
    dataset = EpisodeDataset(index_path, split, require_labels=True)
    settings = MultiResolutionSettings.from_config(config)
    destination = settings.healthy_profile_path
    if destination is None:
        raise DataContractError("model.multiresolution.physics.healthy_profile_path is required")
    profile_settings = replace(
        settings, healthy_profile_path=None, require_healthy_profile=False
    )
    preprocessor = MultiResolutionEpisodePreprocessor(profile_settings)
    builder = HealthyProfileBuilder(preprocessor.physics_extractor.profile_feature_names)
    accepted_episodes = 0
    healthy_indices = [
        index
        for index, record in enumerate(dataset.records)
        if str(record.get("global_class_name") or "") == "normal"
        or (
            not record.get("global_class_name")
            and str(record.get("split_stratum") or "") == "healthy"
        )
    ]
    rejected_nonhealthy = len(dataset) - len(healthy_indices)

    physics_cfg = config["model"]["multiresolution"]["physics"]
    profile_num_workers = int(physics_cfg.get("profile_num_workers", 0))
    if profile_num_workers < 0:
        raise ValueError("profile_num_workers must be non-negative")
    if profile_num_workers:
        from torch.utils.data import DataLoader, Subset

        from ..training.data import MultiResolutionBatchCollator, _configure_worker

        profile_config = copy.deepcopy(config)
        profile_physics = profile_config["model"]["multiresolution"]["physics"]
        profile_physics["healthy_profile_path"] = None
        profile_physics["require_healthy_profile"] = False
        profile_batch_size = max(
            1, int(physics_cfg.get("profile_batch_size", 2))
        )
        loader = DataLoader(
            Subset(dataset, healthy_indices),
            batch_size=profile_batch_size,
            shuffle=False,
            num_workers=profile_num_workers,
            persistent_workers=True,
            prefetch_factor=max(
                1, int(physics_cfg.get("profile_prefetch_factor", 1))
            ),
            collate_fn=MultiResolutionBatchCollator(
                profile_config, retain_prepared=True
            ),
            worker_init_fn=_configure_worker,
        )
        prepared_episodes = (
            prepared
            for batch in loader
            for prepared in batch.require_full_prepared()
        )
    else:
        prepared_episodes = (
            preprocessor.prepare(dataset[index]) for index in healthy_indices
        )

    for prepared in prepared_episodes:
        episode = prepared.episode
        if episode.label is None or episode.label.global_class_name != "normal":
            raise DataContractError(
                "healthy-profile index metadata disagrees with the serialized label"
            )
        for target_index in range(1, 6):
            builder.add(
                prepared.profile_feature_values[target_index],
                prepared.profile_feature_mask[target_index],
                dataset_id=episode.provenance.dataset_id,
                structure_id=episode.provenance.structure_id,
                target_index=target_index,
                split_group=episode.provenance.split_group,
            )
        accepted_episodes += 1
    if accepted_episodes == 0:
        raise DataContractError(f"no wholly healthy episodes were found in split {split!r}")
    profile = builder.finalize(
        minimum_count=int(physics_cfg.get("profile_minimum_count", 8)),
        minimum_independent_groups=int(
            physics_cfg.get("profile_minimum_independent_groups", 2)
        ),
        std_floor=float(physics_cfg.get("profile_std_floor", 1e-3)),
        allow_dataset_fallback=bool(physics_cfg.get("profile_allow_dataset_fallback", True)),
        allow_global_fallback=bool(physics_cfg.get("profile_allow_global_fallback", False)),
        metadata={
            "split": split,
            "fitted_utc": utc_now_iso(),
            "public_data_only": True,
            "healthy_episodes": accepted_episodes,
            "rejected_nonhealthy_episodes": rejected_nonhealthy,
        },
    )
    profile.save(destination)
    exact_groups = {
        key: int(group.get("group_count", 0))
        for key, group in profile.groups.items()
        if "structure:*" not in key and "dataset:*" not in key
    }
    invalid_groups = sorted(
        key for key, group in profile.groups.items() if not bool(group.get("group_valid", True))
    )
    report = {
        "profile": str(destination),
        "feature_count": len(profile.feature_names),
        "group_count": len(profile.groups),
        "healthy_episodes": accepted_episodes,
        "rejected_nonhealthy_episodes": rejected_nonhealthy,
        "num_workers": profile_num_workers,
        "minimum_independent_groups": int(
            physics_cfg.get("profile_minimum_independent_groups", 2)
        ),
        "exact_group_counts": exact_groups,
        "invalid_profile_groups": invalid_groups,
        "global_fallback_enabled": bool(
            physics_cfg.get("profile_allow_global_fallback", False)
        ),
    }
    atomic_write_json(destination.with_suffix(".report.json"), report)
    return report
