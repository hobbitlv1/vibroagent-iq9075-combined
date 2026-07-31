

def pytest_collection_modifyitems(config, items):
    """The four template-provenance tests re-derive the codes_v3 prompt asset
    from the immutable 11k-example training set (data/codes_sft_v3), which is
    not distributed with this repository. Skip them cleanly when it is absent;
    the asset itself is still exercised by every other codes test."""
    import pathlib
    import pytest
    train = pathlib.Path(__file__).resolve().parents[1] / "data" / "codes_sft_v3" / "train.jsonl"
    if train.is_file():
        return
    marker = pytest.mark.skip(reason="data/codes_sft_v3 not distributed (training set)")
    dataset_bound = {
        "test_template_asset_matches_dataset",
        "test_prompt_rebuild_is_byte_exact",
        "test_assistant_labels_within_vocabulary",
        "test_parse_accepts_training_answers",
    }
    for item in items:
        if item.name in dataset_bound:
            item.add_marker(marker)
