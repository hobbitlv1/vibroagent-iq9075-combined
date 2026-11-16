"""The model server turns the NPU backend's host buffers off before GenieX is imported."""

import vibroagent_mcp.vibrogemma_geniex_server as geniex_server
from vibroagent_mcp.vibrogemma_geniex_server import _apply_npu_backend_defaults


def test_host_buffers_are_off_by_default(monkeypatch):
    monkeypatch.delenv("GGML_HEXAGON_HOSTBUF", raising=False)

    _apply_npu_backend_defaults()

    assert geniex_server.os.environ["GGML_HEXAGON_HOSTBUF"] == "0"


def test_an_explicit_setting_is_kept(monkeypatch):
    monkeypatch.setenv("GGML_HEXAGON_HOSTBUF", "1")

    _apply_npu_backend_defaults()

    assert geniex_server.os.environ["GGML_HEXAGON_HOSTBUF"] == "1"


def test_load_applies_the_default_before_importing_geniex(monkeypatch):
    order = []
    runtime = geniex_server.VibroGemmaRuntime.__new__(geniex_server.VibroGemmaRuntime)
    monkeypatch.setattr(geniex_server, "_apply_npu_backend_defaults", lambda: order.append("defaults"))

    class _Importer:
        def find_spec(self, name, path=None, target=None):
            if name == "geniex":
                order.append("geniex import")
                raise ImportError("stop here")
            return None

    monkeypatch.setattr(geniex_server.sys, "meta_path", [_Importer(), *geniex_server.sys.meta_path])
    monkeypatch.delitem(geniex_server.sys.modules, "geniex", raising=False)
    try:
        runtime.load()
    except ImportError:
        pass

    assert order[:2] == ["defaults", "geniex import"]
