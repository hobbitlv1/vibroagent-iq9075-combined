"""Small MCP stdio client for the building-vibration tools."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any


BUILDING_TOOLS = (
    "list_building_sensors",
    "read_building_sensor_window",
    "run_autonomous_sensor_check",
    "run_building_sensor_agent",
    "run_building_vibration_agent_pipeline",
    "export_building_validation_dataset",
)


def _json_default(obj: Any) -> str:
    return str(obj)


def _load_mcp_client_symbols():
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        raise RuntimeError(
            "The MCP SDK is not installed. Install the project with `pip install -e .` "
            "or install `mcp[cli]` before running the MCP client."
        ) from exc
    return ClientSession, StdioServerParameters, stdio_client


def _sensor_ids(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def _tool_arguments(args: argparse.Namespace) -> dict[str, Any]:
    common = {
        "config_path": args.config_path,
        "start_time_s": args.start_time_s,
        "duration_s": args.duration_s,
        "axis": args.axis,
        "require_current": not args.allow_recorded_data,
    }

    if args.tool == "list_building_sensors":
        return {"config_path": args.config_path}
    if args.tool == "read_building_sensor_window":
        return {
            **common,
            "sensor_id": args.sensor_id,
            "include_samples": args.include_samples,
            "max_preview_samples": args.preview_samples,
            "max_return_samples": args.max_return_samples,
        }
    if args.tool == "run_building_sensor_agent":
        return {
            **common,
            "sensor_id": args.sensor_id,
            "baseline_sensor_id": args.baseline_sensor_id,
        }
    if args.tool in {"run_autonomous_sensor_check", "run_building_vibration_agent_pipeline"}:
        return {
            **common,
            "baseline_sensor_id": args.baseline_sensor_id,
            "sensor_ids": _sensor_ids(args.sensor_ids),
        }
    if args.tool == "export_building_validation_dataset":
        return {
            **common,
            "baseline_sensor_id": args.baseline_sensor_id,
            "sensor_ids": _sensor_ids(args.sensor_ids),
            "output_dir": args.output_dir,
        }
    raise ValueError(f"Unsupported tool: {args.tool}")


async def run_client(args: argparse.Namespace) -> None:
    ClientSession, StdioServerParameters, stdio_client = _load_mcp_client_symbols()
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "vibroagent_mcp.mcp_server", "--transport", "stdio"],
        env=dict(os.environ),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            available = [tool.name for tool in tools.tools]
            print("Available MCP tools:", available)
            if args.tool not in available:
                raise RuntimeError(f"Tool {args.tool!r} is not available from the building-monitoring server.")

            result = await session.call_tool(args.tool, arguments=_tool_arguments(args))
            structured = getattr(result, "structuredContent", None)
            if structured is not None:
                print(json.dumps(structured, indent=2, default=_json_default))
            else:
                print(result.content[0].text if result.content else result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Call a building-vibration MCP tool over stdio")
    parser.add_argument("--tool", choices=BUILDING_TOOLS, default="run_autonomous_sensor_check")
    parser.add_argument("--config-path", default="config/sensors.live.yaml")
    parser.add_argument("--baseline-sensor-id", default="baseline")
    parser.add_argument("--sensor-id", default="baseline")
    parser.add_argument("--sensor-ids", default=None, help="Comma-separated target sensor IDs")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--start-time-s", type=float, default=None)
    parser.add_argument("--axis", default=None, choices=[None, "norm", "x", "y", "z", "0", "1", "2"])
    parser.add_argument("--preview-samples", type=int, default=16)
    parser.add_argument("--max-return-samples", type=int, default=200_000)
    parser.add_argument("--include-samples", action="store_true")
    parser.add_argument("--output-dir", default="./validation_exports")
    parser.add_argument(
        "--allow-recorded-data",
        action="store_true",
        help="Allow saved/replay data instead of requiring an actively updating acquisition.",
    )
    args = parser.parse_args()
    try:
        asyncio.run(run_client(args))
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
