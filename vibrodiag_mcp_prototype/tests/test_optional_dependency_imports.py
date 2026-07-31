import importlib.util

import pytest


def test_mcp_modules_import_without_mcp_sdk():
    from vibroagent_mcp import mcp_client_demo, mcp_server

    assert hasattr(mcp_server, "list_building_sensors")
    assert hasattr(mcp_client_demo, "main")

    if importlib.util.find_spec("mcp") is None:
        with pytest.raises(RuntimeError, match="MCP SDK is not installed"):
            mcp_client_demo._load_mcp_client_symbols()
