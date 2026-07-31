"""Building-vibration MCP server.

Run:
    python -m vibroagent_mcp.mcp_server --transport stdio
    python -m vibroagent_mcp.mcp_server --transport streamable-http
"""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import redirect_stdout
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    _MCP_IMPORT_ERROR: ImportError | None = exc

    class FastMCP:  # type: ignore[no-redef]
        """Import-time placeholder used when the optional MCP SDK is absent."""

        def __init__(self, *_args: Any, **_kwargs: Any):
            pass

        def tool(self):
            def decorator(func):
                return func

            return decorator

        def run(self, *_args: Any, **_kwargs: Any):
            raise RuntimeError(
                "The MCP SDK is not installed. Install the project with `pip install -e .` "
                "or install `mcp[cli]` before running the MCP server."
            ) from _MCP_IMPORT_ERROR
else:
    _MCP_IMPORT_ERROR = None

from .service import (
    compare_building_sensor_psd_aligned_service,
    export_building_validation_dataset_service,
    list_building_sensors_service,
    read_building_sensor_psd_service,
    read_building_sensor_window_service,
    run_autonomous_sensor_check_service,
    run_building_sensor_agent_service,
    run_building_vibration_agent_pipeline_service,
)

mcp = FastMCP("Building Vibration Monitoring Server", json_response=True)


def _run_tool_safely(func, *args, **kwargs):
    """Keep MCP stdio stdout reserved for JSON-RPC protocol frames."""
    _redirect_stdatalog_logs_to_stderr()
    with redirect_stdout(sys.stderr):
        try:
            return func(*args, **kwargs)
        finally:
            _redirect_stdatalog_logs_to_stderr()


def _redirect_stdatalog_logs_to_stderr() -> None:
    logger = logging.getLogger("HSDatalogApp")
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stderr)


@mcp.tool()
def list_building_sensors(
    config_path: str = "config/sensors.live.yaml",
) -> dict[str, Any]:
    """List configured building sensors and the baseline/reference sensor."""
    return _run_tool_safely(list_building_sensors_service, config_path=config_path)


@mcp.tool()
def read_building_sensor_window(
    config_path: str = "config/sensors.live.yaml",
    sensor_id: str = "baseline",
    axis: str | None = None,
    start_time_s: float | None = None,
    duration_s: float = 5.0,
    include_samples: bool = False,
    max_preview_samples: int = 16,
    max_return_samples: int = 200_000,
    require_current: bool = True,
) -> dict[str, Any]:
    """Read one registered building sensor and return compact signal statistics.

    The sensor ID is resolved through the building configuration, so the tool
    cannot silently switch to another board or arbitrary acquisition folder.
    """
    return _run_tool_safely(
        read_building_sensor_window_service,
        config_path=config_path,
        sensor_id=sensor_id,
        axis=axis,
        start_time_s=start_time_s,
        duration_s=duration_s,
        include_samples=include_samples,
        max_preview_samples=max_preview_samples,
        max_return_samples=max_return_samples,
        require_current=require_current,
    )


@mcp.tool()
def read_building_sensor_psd(
    config_path: str = "config/sensors.live.yaml",
    sensor_id: str = "baseline",
    axis: str | None = None,
    start_time_s: float | None = None,
    duration_s: float = 10.0,
    estimator: str = "welch",
    require_current: bool = True,
) -> dict[str, Any]:
    """Compact PSD summary for one registered sensor: top peaks, band powers,
    and impact/ring-down status. Numbers only — interpretation is the model's."""
    return _run_tool_safely(
        read_building_sensor_psd_service,
        config_path=config_path,
        sensor_id=sensor_id,
        axis=axis,
        start_time_s=start_time_s,
        duration_s=duration_s,
        estimator=estimator,
        require_current=require_current,
    )


@mcp.tool()
def compare_building_sensor_psd(
    config_path: str = "config/sensors.live.yaml",
    sensor_ids: list[str] | None = None,
    axis: str | None = None,
    duration_s: float = 8.0,
    estimator: str = "welch",
    require_current: bool = True,
) -> dict[str, Any]:
    """PSD summaries of the SAME wall-clock window across several sensors
    (default: all), with per-band and total deltas against the reference
    sensor. Windows are wall-aligned to ~±0.25s (boards free-run; no hardware
    sync). Numbers only — interpretation is the model's."""
    return _run_tool_safely(
        compare_building_sensor_psd_aligned_service,
        config_path=config_path,
        sensor_ids=sensor_ids,
        axis=axis,
        duration_s=duration_s,
        estimator=estimator,
        require_current=require_current,
    )


@mcp.tool()
def run_autonomous_sensor_check(
    config_path: str = "config/sensors.live.yaml",
    baseline_sensor_id: str = "baseline",
    sensor_ids: list[str] | str | None = None,
    start_time_s: float | None = None,
    duration_s: float = 10.0,
    axis: str | None = None,
    require_current: bool = True,
) -> dict[str, Any]:
    """Check all selected building sensors relative to the reference sensor."""
    return _run_tool_safely(
        run_autonomous_sensor_check_service,
        config_path=config_path,
        baseline_sensor_id=baseline_sensor_id,
        sensor_ids=sensor_ids,
        start_time_s=start_time_s,
        duration_s=duration_s,
        axis=axis,
        require_current=require_current,
    )


@mcp.tool()
def run_building_sensor_agent(
    config_path: str = "config/sensors.live.yaml",
    sensor_id: str = "",
    baseline_sensor_id: str = "baseline",
    start_time_s: float | None = None,
    duration_s: float = 10.0,
    axis: str | None = None,
    require_current: bool = True,
) -> dict[str, Any]:
    """Assess one target building sensor relative to the reference sensor."""
    return _run_tool_safely(
        run_building_sensor_agent_service,
        config_path=config_path,
        sensor_id=sensor_id,
        baseline_sensor_id=baseline_sensor_id,
        start_time_s=start_time_s,
        duration_s=duration_s,
        axis=axis,
        require_current=require_current,
    )


@mcp.tool()
def run_building_vibration_agent_pipeline(
    config_path: str = "config/sensors.live.yaml",
    baseline_sensor_id: str = "baseline",
    sensor_ids: list[str] | str | None = None,
    start_time_s: float | None = None,
    duration_s: float = 10.0,
    axis: str | None = None,
    require_current: bool = True,
) -> dict[str, Any]:
    """Run the multi-sensor building-vibration relative-monitoring pipeline."""
    return _run_tool_safely(
        run_building_vibration_agent_pipeline_service,
        config_path=config_path,
        baseline_sensor_id=baseline_sensor_id,
        sensor_ids=sensor_ids,
        start_time_s=start_time_s,
        duration_s=duration_s,
        axis=axis,
        require_current=require_current,
    )


@mcp.tool()
def export_building_validation_dataset(
    config_path: str = "config/sensors.live.yaml",
    baseline_sensor_id: str = "baseline",
    sensor_ids: list[str] | str | None = None,
    start_time_s: float | None = None,
    duration_s: float = 10.0,
    axis: str | None = None,
    output_dir: str = "./validation_exports",
    require_current: bool = True,
) -> dict[str, Any]:
    """Run the building pipeline and write MATLAB validation CSV/JSONL artifacts."""
    return _run_tool_safely(
        export_building_validation_dataset_service,
        config_path=config_path,
        baseline_sensor_id=baseline_sensor_id,
        sensor_ids=sensor_ids,
        start_time_s=start_time_s,
        duration_s=duration_s,
        axis=axis,
        output_dir=output_dir,
        require_current=require_current,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Building Vibration Monitoring MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP transport. Use stdio for local clients; streamable-http for network/MCP Inspector.",
    )
    args = parser.parse_args()
    try:
        mcp.run(transport=args.transport)
    except RuntimeError as exc:
        if _MCP_IMPORT_ERROR is not None:
            parser.exit(1, f"{exc}\n")
        raise


if __name__ == "__main__":
    main()
