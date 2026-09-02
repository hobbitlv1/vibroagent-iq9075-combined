"""Public-dataset contracts, deterministic episode construction, and caches.

Imports are lazy so small geometry/manifest modules can be used by simulation code
without recursively importing the episode builder and its simulation wrappers.
"""

from typing import Any

__all__ = [
    "DatasetManifest",
    "EpisodeDataset",
    "PublicEpisodeBuilder",
    "SplitRatios",
    "assign_group",
    "assert_disjoint_groups",
    "load_dataset_manifests",
    "load_episode",
    "save_episode",
]


def __getattr__(name: str) -> Any:
    if name in {"EpisodeDataset", "PublicEpisodeBuilder"}:
        from .episodes import EpisodeDataset, PublicEpisodeBuilder

        return {
            "EpisodeDataset": EpisodeDataset,
            "PublicEpisodeBuilder": PublicEpisodeBuilder,
        }[name]
    if name in {"DatasetManifest", "load_dataset_manifests"}:
        from .manifest import DatasetManifest, load_dataset_manifests

        return {
            "DatasetManifest": DatasetManifest,
            "load_dataset_manifests": load_dataset_manifests,
        }[name]
    if name in {"load_episode", "save_episode"}:
        from .serialization import load_episode, save_episode

        return {"load_episode": load_episode, "save_episode": save_episode}[name]
    if name in {"SplitRatios", "assign_group", "assert_disjoint_groups"}:
        from .splits import SplitRatios, assign_group, assert_disjoint_groups

        return {
            "SplitRatios": SplitRatios,
            "assign_group": assign_group,
            "assert_disjoint_groups": assert_disjoint_groups,
        }[name]
    raise AttributeError(name)
