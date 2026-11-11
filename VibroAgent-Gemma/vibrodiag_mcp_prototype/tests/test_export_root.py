"""The model-callable export tool may only write under the export root."""

import pytest

from vibroagent_mcp.service import _resolve_export_output_dir, export_building_validation_dataset_service


def test_export_output_dir_is_pinned_to_export_root(tmp_path, monkeypatch):
    root = tmp_path / "export_root"
    monkeypatch.setenv("VIBRO_EXPORT_ROOT", str(root))

    assert _resolve_export_output_dir(".") == root.resolve()
    assert _resolve_export_output_dir(None) == root.resolve()
    assert _resolve_export_output_dir("run_7/sub") == (root / "run_7" / "sub").resolve()
    assert _resolve_export_output_dir(str(root / "abs")) == (root / "abs").resolve()

    for escape in ("../elsewhere", str(tmp_path / "elsewhere"), "/etc", "~/.ssh", "run/../../x"):
        with pytest.raises(ValueError, match="export root"):
            _resolve_export_output_dir(escape)


def test_export_service_refuses_escape_before_running_the_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBRO_EXPORT_ROOT", str(tmp_path / "export_root"))
    result = export_building_validation_dataset_service(
        config_path="config/does-not-matter.yaml",
        output_dir=str(tmp_path / "outside"),
        require_current=False,
    )
    assert result["status"] == "error"
    assert result["error"] == "export_dir_not_allowed"
    assert result["requested_output_dir"] == str(tmp_path / "outside")
    assert not (tmp_path / "outside").exists()
