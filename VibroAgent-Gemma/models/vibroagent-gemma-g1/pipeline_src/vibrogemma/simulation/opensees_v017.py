from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


class OpenSeesRunError(RuntimeError):
    """Raised when a pinned OpenSees simulation does not complete as declared."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TwinParameters:
    """Parameters for a dynamic-reference plus five-target reduced-order twin.

    The model is intentionally a dataset-conditioned surrogate, not a claim of a
    geometrically exact publisher finite-element model.  Node 1 is a dynamic
    context/reference mass; Nodes 2..6 are the five permanent target zones.
    """

    masses: tuple[float, float, float, float, float, float]
    stiffnesses: tuple[float, float, float, float, float, float]
    damping_ratio: float
    damage_fraction: float
    dt: float
    steps: int

    def validate(self) -> None:
        if len(self.masses) != 6 or len(self.stiffnesses) != 6:
            raise ValueError("TwinParameters requires exactly six masses and springs")
        if any(value <= 0 for value in self.masses):
            raise ValueError("all masses must be positive")
        if any(value <= 0 for value in self.stiffnesses):
            raise ValueError("all stiffnesses must be positive")
        if not 0.0 <= self.damping_ratio <= 0.2:
            raise ValueError("damping_ratio must be in [0, 0.2]")
        if not 0.0 <= self.damage_fraction < 0.95:
            raise ValueError("damage_fraction must be in [0, 0.95)")
        if self.dt <= 0 or self.steps < 16:
            raise ValueError("invalid transient discretization")


def _tcl_list(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def build_tcl_model(
    *,
    parameters: TwinParameters,
    force_path: Path,
    output_path: Path,
    damage_target: int | None,
) -> str:
    """Create a deterministic Tcl analysis for one counterfactual state.

    Damage target 1..5 reduces the spring directly feeding target node 2..6.
    The context/reference node remains dynamic and is excited by a measured-like
    force.  No restrained zero-acceleration node is exposed as a board signal.
    """

    parameters.validate()
    if damage_target is not None and damage_target not in {1, 2, 3, 4, 5}:
        raise ValueError("damage_target must be None or one of 1..5")

    stiffnesses = list(parameters.stiffnesses)
    if damage_target is not None:
        spring_index = int(damage_target)  # spring 0 feeds ref; 1..5 feed targets
        stiffnesses[spring_index] *= 1.0 - parameters.damage_fraction

    # Mass-proportional plus initial-stiffness damping.  The simple closed-form
    # coefficients anchor damping around two representative angular frequencies.
    # Counterfactual states must share the healthy damping law; changing Rayleigh
    # coefficients with the damaged spring would confound damage and damping.
    k_med = float(np.median(parameters.stiffnesses))
    m_med = float(np.median(parameters.masses))
    omega_1 = max(math.sqrt(k_med / m_med) * 0.45, 1e-3)
    omega_2 = max(math.sqrt(k_med / m_med) * 2.0, omega_1 * 1.1)
    zeta = parameters.damping_ratio
    alpha_m = 2.0 * zeta * omega_1 * omega_2 / (omega_1 + omega_2)
    beta_k = 2.0 * zeta / (omega_1 + omega_2)

    lines = [
        "wipe",
        "model BasicBuilder -ndm 1 -ndf 1",
        "node 0 0.0",
        "fix 0 1",
    ]
    for node, mass in enumerate(parameters.masses, start=1):
        lines.extend([f"node {node} 0.0", f"mass {node} {mass:.17g}"])
    for spring_index, stiffness in enumerate(stiffnesses, start=1):
        left = spring_index - 1
        right = spring_index
        material = 100 + spring_index
        element = 200 + spring_index
        lines.extend(
            [
                f"uniaxialMaterial Elastic {material} {stiffness:.17g}",
                f"element zeroLength {element} {left} {right} -mat {material} -dir 1",
            ]
        )
    lines.extend(
        [
            f"rayleigh {alpha_m:.17g} 0.0 {beta_k:.17g} 0.0",
            f'timeSeries Path 1 -dt {parameters.dt:.17g} -filePath "{force_path.as_posix()}" -factor 1.0',
            "pattern Plain 1 1 { load 1 1.0 }",
            f'recorder Node -file "{output_path.as_posix()}" -time -node 1 2 3 4 5 6 -dof 1 accel',
            "constraints Plain",
            "numberer RCM",
            "system BandGeneral",
            "test NormDispIncr 1.0e-10 40 0",
            "algorithm Newton",
            "integrator Newmark 0.5 0.25",
            "analysis Transient",
            f"set ok [analyze {int(parameters.steps)} {parameters.dt:.17g}]",
            'puts stdout "VIBROGEMMA_ANALYZE_STATUS=$ok"',
            "wipe",
            "exit",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_node_recorder(path: Path, expected_steps: int) -> np.ndarray:
    if not path.is_file():
        raise OpenSeesRunError(f"OpenSees did not create recorder output: {path}")
    try:
        matrix = np.loadtxt(path, dtype=np.float64)
    except Exception as exc:  # pragma: no cover - defensive context
        raise OpenSeesRunError(f"cannot parse OpenSees recorder output {path}: {exc}") from exc
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[1] != 7:
        raise OpenSeesRunError(
            f"expected time plus six accelerations, observed shape={matrix.shape}"
        )
    if matrix.shape[0] < max(16, expected_steps - 2):
        raise OpenSeesRunError(
            f"incomplete OpenSees transient: expected≈{expected_steps}, observed={matrix.shape[0]}"
        )
    values = matrix[:, 1:].T
    if not np.isfinite(values).all():
        raise OpenSeesRunError("OpenSees output contains non-finite values")
    return values


def run_counterfactual_family(
    *,
    binary: Path,
    work_dir: Path,
    force: np.ndarray,
    parameters: TwinParameters,
    timeout_seconds: int = 300,
) -> dict[str, np.ndarray]:
    """Run healthy plus five single-zone counterfactual states.

    Returns six arrays with shape [6, time].  Every state uses the same sampled
    model and excitation; only the declared zone stiffness changes.
    """

    parameters.validate()
    binary = binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"OpenSees binary is not executable: {binary}")
    work_dir.mkdir(parents=True, exist_ok=True)
    force = np.asarray(force, dtype=np.float64).reshape(-1)
    if force.size != parameters.steps:
        raise ValueError(f"force length {force.size} != declared steps {parameters.steps}")
    if not np.isfinite(force).all():
        raise ValueError("force contains non-finite values")
    force_path = work_dir / "force.txt"
    np.savetxt(force_path, force, fmt="%.17g")

    outputs: dict[str, np.ndarray] = {}
    receipts: dict[str, dict[str, object]] = {}
    for damage_target in (None, 1, 2, 3, 4, 5):
        state = "healthy" if damage_target is None else f"target_{damage_target}"
        output_path = work_dir / f"{state}.out"
        script_path = work_dir / f"{state}.tcl"
        script_path.write_text(
            build_tcl_model(
                parameters=parameters,
                force_path=force_path,
                output_path=output_path,
                damage_target=damage_target,
            ),
            encoding="utf-8",
        )
        run_env = os.environ.copy()
        tcl_runtime = binary.resolve().parent.parent / "lib" / "tcl8.6"
        if (tcl_runtime / "init.tcl").is_file():
            run_env["TCL_LIBRARY"] = str(tcl_runtime)
        completed = subprocess.run(
            [str(binary), str(script_path)],
            cwd=work_dir,
            env=run_env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        log_path = work_dir / f"{state}.log"
        log_path.write_text(
            completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
            encoding="utf-8",
        )
        status_matches = re.findall(r"VIBROGEMMA_ANALYZE_STATUS=(-?\d+)", completed.stdout)
        status = int(status_matches[-1]) if status_matches else None
        if completed.returncode != 0 or status != 0:
            raise OpenSeesRunError(
                f"OpenSees state {state} failed: returncode={completed.returncode}, "
                f"analysis_status={status}; see {log_path}"
            )
        outputs[state] = _parse_node_recorder(output_path, parameters.steps)
        receipts[state] = {
            "script_sha256": sha256_file(script_path),
            "output_sha256": sha256_file(output_path),
            "log": str(log_path),
            "damage_target": damage_target,
        }

    (work_dir / "counterfactual_receipt.json").write_text(
        json.dumps(
            {
                "schema": "vibrogemma-opensees-counterfactual-v0.17.0",
                "binary": str(binary),
                "binary_sha256": sha256_file(binary),
                "parameters": asdict(parameters),
                "force_sha256": sha256_file(force_path),
                "states": receipts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return outputs
