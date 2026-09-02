from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.pretty import Pretty

from .config import load_config, resolve_from_config
from .utils import atomic_write_json, utc_now_iso

app = typer.Typer(no_args_is_help=True, help="Gemma-only six-board vibration-classification pipeline")
datasets_app = typer.Typer(no_args_is_help=True, help="Public dataset verification and episode construction")
app.add_typer(datasets_app, name="datasets")
console = Console()

ConfigPath = Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)]


def _show(value: Any) -> None:
    console.print(Pretty(value, expand_all=True))


@datasets_app.command("verify")
def datasets_verify(
    config: ConfigPath = Path("configs/train_public.yaml"),
    require_licence_markers: Annotated[bool, typer.Option("--require-licences/--skip-licence-markers")] = True,
) -> None:
    from .data.episodes import PublicEpisodeBuilder

    payload = load_config(config)
    _show(PublicEpisodeBuilder(payload).verify(require_licence_markers=require_licence_markers))


@datasets_app.command("inspect")
def datasets_inspect(
    config: ConfigPath = Path("configs/train_public.yaml"),
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    from .data.episodes import PublicEpisodeBuilder

    payload = load_config(config)
    report = PublicEpisodeBuilder(payload).inspect()
    destination = output or resolve_from_config(payload, "outputs/dataset_inspection.json")
    atomic_write_json(destination, report)
    _show({"output": str(destination), "files": len(report)})


@datasets_app.command("accept-licence")
def datasets_accept_licence(
    dataset_id: Annotated[str, typer.Argument(help="Exact dataset ID from configs/datasets")],
    config: ConfigPath = Path("configs/train_public.yaml"),
    acknowledged: Annotated[
        bool,
        typer.Option("--i-reviewed-the-source-licence", help="Required explicit acknowledgement"),
    ] = False,
) -> None:
    from .data.manifest import load_dataset_manifests

    if not acknowledged:
        raise typer.BadParameter("Pass --i-reviewed-the-source-licence after reviewing the source terms")
    payload = load_config(config)
    matches = [manifest for manifest in load_dataset_manifests(payload) if manifest.id == dataset_id]
    if len(matches) != 1:
        raise typer.BadParameter(f"Unknown dataset ID: {dataset_id}")
    manifest = matches[0]
    marker = manifest.licence_marker(payload)
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        marker,
        {
            "dataset_id": manifest.id,
            "source_page": manifest.source_page,
            "licence_text_from_manifest": manifest.licence,
            "acknowledged_utc": utc_now_iso(),
        },
    )
    _show({"created": str(marker)})


@datasets_app.command("make-index-template")
def datasets_make_index_template(
    dataset_id: Annotated[str, typer.Argument()],
    config: ConfigPath = Path("configs/train_public.yaml"),
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    from .data.indexing import write_index_template
    from .data.manifest import load_dataset_manifests

    payload = load_config(config)
    matches = [manifest for manifest in load_dataset_manifests(payload) if manifest.id == dataset_id]
    if len(matches) != 1:
        raise typer.BadParameter(f"Unknown dataset ID: {dataset_id}")
    manifest = matches[0]
    destination = output or (manifest.raw_root(payload) / "index.template.jsonl")
    write_index_template(manifest, payload, destination)
    _show({"output": str(destination), "dataset_id": dataset_id})


@datasets_app.command("auto-index")
def datasets_auto_index(
    dataset_id: Annotated[
        str,
        typer.Argument(help="Exact selected dataset ID, or 'all' for every supported selected manifest"),
    ] = "all",
    config: ConfigPath = Path("configs/train_public.yaml"),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Generate strict indexes from version-pinned publisher schemas without an upload step."""

    from .data.auto_index import auto_index_all, auto_index_dataset

    payload = load_config(config)
    if dataset_id.lower() == "all":
        _show(auto_index_all(payload, overwrite=overwrite))
    else:
        _show(auto_index_dataset(payload, dataset_id, overwrite=overwrite))


@datasets_app.command("prepare-private-drive")
def datasets_prepare_private_drive(
    folder_id: Annotated[str, typer.Option("--folder-id", help="Raw Google Drive folder ID")],
    datasets: Annotated[
        list[str],
        typer.Option("--dataset", help="Repeat for every checksum-locked dataset to prepare"),
    ],
    config: ConfigPath = Path("configs/train_private_drive.yaml"),
    bundle: Annotated[Path, typer.Option("--bundle", exists=True, dir_okay=False)] = Path(
        "configs/private_drive_bundle.yaml"
    ),
    archive_cache_root: Annotated[Path | None, typer.Option("--archive-cache-root")] = None,
    acknowledge_open_licences: Annotated[bool, typer.Option("--acknowledge-open-licences")] = False,
    acknowledge_z24_private_terms: Annotated[
        bool,
        typer.Option("--acknowledge-z24-private-terms"),
    ] = False,
    acknowledge_qugs_private_rights: Annotated[
        bool,
        typer.Option("--acknowledge-qugs-private-rights"),
    ] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    keep_archives: Annotated[bool, typer.Option("--keep-archives")] = False,
    minimum_free_gib: Annotated[float, typer.Option("--minimum-free-gib")] = 8.0,
) -> None:
    """Prepare private Drive bundles without embedding credentials or raw data."""

    from .data.auto_index import auto_index_dataset
    from .data.manifest import load_dataset_manifests
    from .data.private_drive import load_private_drive_bundle, prepare_private_drive_bundle

    payload = load_config(config)
    public_root = resolve_from_config(payload, payload["data"]["public_root"])
    cache_root = archive_cache_root or (public_root.parent / "private_drive_archives")
    report = prepare_private_drive_bundle(
        bundle=load_private_drive_bundle(bundle),
        folder_id=folder_id,
        selected_dataset_ids=datasets,
        public_root=public_root,
        archive_cache_root=cache_root,
        acknowledge_open_licences=acknowledge_open_licences,
        acknowledge_z24_private_terms=acknowledge_z24_private_terms,
        acknowledge_qugs_private_rights=acknowledge_qugs_private_rights,
        overwrite=overwrite,
        delete_archives=not keep_archives,
        minimum_free_gib=minimum_free_gib,
    )
    manifests = {manifest.id: manifest for manifest in load_dataset_manifests(payload)}
    strict_indexes = []
    for dataset_id in datasets:
        manifest = manifests.get(dataset_id)
        if manifest is None or not manifest.training_allowed:
            continue
        index_path = manifest.index_path(payload)
        report_path = manifest.raw_root(payload) / "auto_index_report.json"
        if index_path.exists() and report_path.exists() and not overwrite:
            strict_indexes.append(
                {
                    "dataset_id": dataset_id,
                    "index": str(index_path),
                    "entries": sum(1 for line in index_path.open("r", encoding="utf-8") if line.strip()),
                    "report": str(report_path),
                    "reused": True,
                }
            )
        else:
            strict_indexes.append(
                auto_index_dataset(
                    payload,
                    dataset_id,
                    overwrite=overwrite or index_path.exists(),
                )
            )
    report["strict_indexes"] = strict_indexes
    _show(report)


@datasets_app.command("build")
def datasets_build(
    config: ConfigPath = Path("configs/train_public.yaml"),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    from .data.episodes import PublicEpisodeBuilder

    _show(PublicEpisodeBuilder(load_config(config)).build(overwrite=overwrite))


@datasets_app.command("audit-splits")
def datasets_audit_splits(
    config: ConfigPath = Path("configs/train_public.yaml"),
) -> None:
    """Fail when a dataset/state stratum is avoidably absent after materialisation."""

    from .data.split_audit import audit_episode_split

    _show(audit_episode_split(load_config(config)))


@datasets_app.command("fit-healthy-profile")
def datasets_fit_healthy_profile(
    config: ConfigPath = Path("configs/train_public.yaml"),
    split: Annotated[str, typer.Option("--split")] = "train",
) -> None:
    from .data.profiles import fit_healthy_profile

    _show(fit_healthy_profile(load_config(config), split=split))


@datasets_app.command("benchmark-preprocessing")
def datasets_benchmark_preprocessing(
    config: ConfigPath = Path("configs/train_public.yaml"),
    split: Annotated[str, typer.Option("--split")] = "train",
    stage: Annotated[str, typer.Option("--stage")] = "pretrain",
    batches: Annotated[int, typer.Option("--batches")] = 4,
) -> None:
    """Benchmark the real parallel NumPy/SciPy preparation DataLoader."""

    from .training.data import benchmark_preprocessing_loader

    if stage not in {"pretrain", "align", "lora"}:
        raise typer.BadParameter("--stage must be pretrain, align, or lora")
    _show(
        benchmark_preprocessing_loader(
            load_config(config),
            split=split,
            stage=stage,
            batches=batches,
        )
    )


@app.command("train-pretrain")
def train_pretrain(
    config: ConfigPath = Path("configs/train_public.yaml"),
    resume: Annotated[Path | None, typer.Option("--resume", exists=True, file_okay=False)] = None,
) -> None:
    from .training.pretrain import run_pretraining

    _show(run_pretraining(load_config(config), resume=str(resume) if resume else None))


@app.command("train-align")
def train_align(
    config: ConfigPath = Path("configs/train_public.yaml"),
    resume: Annotated[Path | None, typer.Option("--resume", exists=True, file_okay=False)] = None,
    initialize_from: Annotated[
        Path | None, typer.Option("--initialize-from", exists=True, file_okay=False)
    ] = None,
) -> None:
    from .training.lm_train import run_alignment_training

    if resume is not None and initialize_from is not None:
        raise typer.BadParameter("--resume and --initialize-from are mutually exclusive")
    _show(
        run_alignment_training(
            load_config(config),
            resume=str(resume) if resume else None,
            initialize_from=str(initialize_from) if initialize_from else None,
        )
    )


@app.command("train-lora")
def train_lora(
    config: ConfigPath = Path("configs/train_public.yaml"),
    resume: Annotated[Path | None, typer.Option("--resume", exists=True, file_okay=False)] = None,
    initialize_from: Annotated[
        Path | None, typer.Option("--initialize-from", exists=True, file_okay=False)
    ] = None,
) -> None:
    from .training.lm_train import run_lora_training

    if resume is not None and initialize_from is not None:
        raise typer.BadParameter("--resume and --initialize-from are mutually exclusive")
    _show(
        run_lora_training(
            load_config(config),
            resume=str(resume) if resume else None,
            initialize_from=str(initialize_from) if initialize_from else None,
        )
    )


@app.command("train-residual-adapter")
def train_residual_adapter(
    champion: Annotated[Path, typer.Option("--champion", exists=True, file_okay=False)],
    config: ConfigPath = Path("configs/train_public.yaml"),
    resume: Annotated[Path | None, typer.Option("--resume", exists=True, file_okay=False)] = None,
) -> None:
    """Train only the zero-gated common-mode residual adapter."""

    from .training.residual_adapter import run_common_mode_residual_training

    _show(
        run_common_mode_residual_training(
            load_config(config),
            champion=champion,
            resume=str(resume) if resume else None,
        )
    )


@app.command("evaluate")
def evaluate(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, file_okay=False)],
    config: ConfigPath = Path("configs/train_public.yaml"),
    split: Annotated[str, typer.Option("--split")] = "test",
    overwrite: Annotated[bool, typer.Option("--overwrite/--resume-existing")] = False,
    include_explanations: Annotated[
        bool, typer.Option("--with-explanations/--record-only")
    ] = True,
) -> None:
    """Evaluate a checkpoint with restart-safe prediction materialisation."""

    from .evaluation.evaluator import evaluate_checkpoint

    _show(
        evaluate_checkpoint(
            load_config(config),
            checkpoint=checkpoint,
            split=split,
            overwrite=overwrite,
            include_explanations=include_explanations,
        )
    )


@app.command("select-checkpoint")
def select_checkpoint(
    stage: Annotated[str, typer.Option("--stage")],
    config: ConfigPath = Path("configs/train_public.yaml"),
    split: Annotated[str, typer.Option("--split")] = "validation",
    episodes_per_stratum: Annotated[int, typer.Option("--episodes-per-stratum")] = 32,
    overwrite: Annotated[bool, typer.Option("--overwrite/--resume-existing")] = False,
) -> None:
    """Select an epoch by balanced operational metrics, not language-model loss."""

    from .evaluation.selection import select_operational_checkpoint

    _show(
        select_operational_checkpoint(
            load_config(config),
            stage=stage,
            split=split,
            episodes_per_stratum=episodes_per_stratum,
            overwrite=overwrite,
        )
    )


@app.command("compare-checkpoints")
def compare_checkpoints(
    candidates: Annotated[Path, typer.Option("--candidates", exists=True, dir_okay=False)],
    config: ConfigPath = Path("configs/train_public.yaml"),
    split: Annotated[str, typer.Option("--split")] = "validation",
    stage1_episodes_per_stratum: Annotated[
        int, typer.Option("--stage1-episodes-per-stratum")
    ] = 32,
    stage2_episodes_per_stratum: Annotated[
        int, typer.Option("--stage2-episodes-per-stratum")
    ] = 128,
    shortlist_challengers: Annotated[
        int, typer.Option("--shortlist-challengers")
    ] = 2,
    overwrite: Annotated[
        bool, typer.Option("--overwrite/--resume-existing")
    ] = False,
) -> None:
    """Compare challengers against an immutable validation champion."""

    from .evaluation.regression import compare_candidate_checkpoints

    _show(
        compare_candidate_checkpoints(
            load_config(config),
            candidate_manifest=candidates,
            split=split,
            stage1_episodes_per_stratum=stage1_episodes_per_stratum,
            stage2_episodes_per_stratum=stage2_episodes_per_stratum,
            shortlist_challengers=shortlist_challengers,
            overwrite=overwrite,
        )
    )


@app.command("validate-selection")
def validate_selection(
    report: Annotated[Path, typer.Option("--report", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    config: ConfigPath = Path("configs/train_public.yaml"),
) -> None:
    """Fail fast when the selected checkpoint still shows state/dataset collapse."""

    from .evaluation.gates import validate_selection_report_file

    payload = load_config(config)
    checkpoint_selection = dict(payload.get("evaluation", {}).get("checkpoint_selection") or {})
    thresholds = dict(checkpoint_selection.get("preflight_gates") or {})
    required = {
        "overall_macro_f1": 0.72,
        "overall_normal_recall": 0.70,
        "overall_anomaly_recall": 0.70,
        "overall_affected_f1": 0.70,
        "minimum_dataset_macro_f1": 0.50,
        "minimum_dataset_state_recall": 0.50,
    }
    required.update(thresholds)
    validation_thresholds = dict(payload.get("evaluation", {}).get("validation_gates") or {})
    if "required_datasets" not in required and validation_thresholds.get("required_datasets"):
        required["required_datasets"] = list(validation_thresholds["required_datasets"])
    result = validate_selection_report_file(report, output=output, thresholds=required)
    _show(result)
    if not result["passed"]:
        raise typer.Exit(code=2)


@app.command("validate-gates")
def validate_gates(
    metrics: Annotated[Path, typer.Option("--metrics", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    config: ConfigPath = Path("configs/train_public.yaml"),
) -> None:
    """Block test evaluation when validation performance is not credible."""

    from .evaluation.gates import validate_metrics_file

    payload = load_config(config)
    thresholds = dict(payload.get("evaluation", {}).get("validation_gates") or {})
    required = {
        "overall_macro_f1": 0.80,
        "overall_normal_recall": 0.90,
        "overall_anomaly_recall": 0.90,
        "overall_affected_f1": 0.80,
        "minimum_dataset_macro_f1": 0.65,
        "minimum_dataset_state_recall": 0.65,
        "maximum_false_alert_events_per_building_day": 0.5,
        "minimum_healthy_hours_for_false_alert_gate": 24.0,
    }
    required.update(thresholds)
    final_protocol = dict(payload.get("final_protocol") or {})
    if "frozen_test_authorized" in final_protocol:
        required["frozen_test_authorized"] = bool(
            final_protocol["frozen_test_authorized"]
        )
    report = validate_metrics_file(metrics, output=output, thresholds=required)
    _show(report)
    if not report["passed"]:
        raise typer.Exit(code=2)


@app.command("calibrate-confidence")
def calibrate_confidence(
    predictions: Annotated[Path, typer.Option("--predictions", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    minimum_auroc: Annotated[float, typer.Option("--minimum-auroc")] = 0.60,
    minimum_errors: Annotated[int, typer.Option("--minimum-errors")] = 20,
) -> None:
    """Fit a validation-only confidence calibrator and operational-use gate."""

    from .model.calibration import fit_confidence_calibrator

    _show(
        fit_confidence_calibrator(
            predictions,
            output,
            minimum_auroc=minimum_auroc,
            minimum_errors=minimum_errors,
        )
    )


@app.command("live")
def live(config: ConfigPath = Path("configs/runtime.yaml")) -> None:
    from .live.coordinator import LiveCoordinator

    coordinator = LiveCoordinator.from_config(load_config(config))
    try:
        coordinator.run_forever()
    except KeyboardInterrupt:
        coordinator.stop()


@app.command("replay")
def replay(
    input: Annotated[Path, typer.Option("--input", "-i", exists=True, dir_okay=False)],
    config: ConfigPath = Path("configs/runtime.yaml"),
    real_time: Annotated[bool, typer.Option("--real-time/--fast")] = True,
) -> None:
    from .live.coordinator import LiveCoordinator

    coordinator = LiveCoordinator.from_config(load_config(config), replay_path=input, replay_real_time=real_time)
    try:
        coordinator.run_replay_until_complete()
    except KeyboardInterrupt:
        coordinator.stop()
    _show(coordinator.last_result or {"status": "no complete window"})


@app.command("serve")
def serve(config: ConfigPath = Path("configs/runtime.yaml")) -> None:
    import uvicorn

    from .inference.service import create_app
    from .inference.store import EventStore

    payload = load_config(config)
    service = payload["service"]
    store = EventStore(resolve_from_config(payload, service["database"]))
    fastapi_app = create_app(store, resolve_from_config(payload, service["dashboard_dir"]))
    uvicorn.run(fastapi_app, host=str(service["host"]), port=int(service["port"]))


@app.command("export")
def export(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, file_okay=False)],
    config: ConfigPath = Path("configs/train_public.yaml"),
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    merge_lora: Annotated[bool, typer.Option("--merge-lora/--keep-adapter")] = True,
    onnx: Annotated[bool, typer.Option("--onnx/--no-onnx")] = True,
) -> None:
    from .export.bundle import export_deployment_bundle

    _show(
        export_deployment_bundle(
            load_config(config),
            checkpoint=checkpoint,
            output_dir=output_dir,
            merge_lora=merge_lora,
            export_onnx=onnx,
        )
    )


if __name__ == "__main__":
    app()
