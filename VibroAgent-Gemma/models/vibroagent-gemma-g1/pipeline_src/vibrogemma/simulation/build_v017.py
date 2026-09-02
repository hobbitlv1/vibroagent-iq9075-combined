from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


_BUILD_SCHEMA = "vibrogemma-opensees-source-build-v0.17.3"
_CMAKE_PATCH_MARKER = "VIBROGEMMA_OPENSEES_380_CMAKE_COMPAT_V0172"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tail_text(path: Path, lines: int = 120) -> str:
    if not path.is_file():
        return f"<missing log: {path}>"
    text = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.splitlines()[-max(1, int(lines)):])


def _run_logged(
    command: Sequence[str],
    log_path: Path,
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    append: bool = False,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as stream:
        stream.write("$ " + " ".join(str(value) for value in command) + "\n")
        stream.flush()
        run_env = os.environ.copy()
        if env is not None:
            run_env.update({str(key): str(value) for key, value in env.items()})
        completed = subprocess.run(
            [str(value) for value in command],
            cwd=cwd,
            env=run_env,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with exit status {completed.returncode}: "
            f"{' '.join(str(value) for value in command)}\n"
            f"Log: {log_path}\n--- log tail ---\n{_tail_text(log_path)}"
        )
    return completed


def discover_source_root(extract_root: Path) -> Path:
    candidates = [extract_root, *[path for path in extract_root.iterdir() if path.is_dir()]]
    for candidate in candidates:
        if (candidate / "CMakeLists.txt").is_file() and (candidate / "SRC").is_dir():
            return candidate
    raise FileNotFoundError(f"cannot find OpenSees source root under {extract_root}")


def discover_binary(build_root: Path) -> Path:
    candidates = [
        path
        for path in build_root.rglob("OpenSees")
        if path.is_file() and os.access(path, os.X_OK)
    ]
    if not candidates:
        raise FileNotFoundError(f"OpenSees executable not found under {build_root}")
    candidates.sort(key=lambda path: (len(path.parts), str(path)))
    return candidates[0]


def _patch_opensees_380_cmake(source_root: Path) -> dict[str, object]:
    """Apply a build-system-only compatibility patch to the extracted v3.8.0 tree.

    OpenSees 3.8.0 mixes the Conan target spellings ``tcl::tcl`` and
    ``TCL::TCL``.  Under some Conan/CMake combinations that sends configuration
    through the system-Tcl branch and attempts to copy an invalid include-relative
    Tcl library path.  The patch normalizes both target spellings and makes Tcl
    runtime staging conditional.  No OpenSees solver/model source is changed.
    """

    cmake_path = source_root / "CMakeLists.txt"
    original = cmake_path.read_text(encoding="utf-8")
    if _CMAKE_PATCH_MARKER in original:
        return {
            "applied": False,
            "already_present": True,
            "path": str(cmake_path),
            "sha256": sha256_file(cmake_path),
        }

    anchor = "find_package(TCL REQUIRED)\n"
    if anchor not in original:
        raise RuntimeError("OpenSees CMakeLists.txt lacks the expected TCL find_package anchor")
    normalization = f'''find_package(TCL REQUIRED)\n\n# {_CMAKE_PATCH_MARKER}\n# Normalize Conan 2 and system FindTCL imported-target spellings.\nif(TARGET TCL::TCL AND NOT TARGET tcl::tcl)\n    add_library(tcl::tcl ALIAS TCL::TCL)\nendif()\nif(TARGET tcl::tcl AND NOT TARGET TCL::TCL)\n    add_library(TCL::TCL ALIAS tcl::tcl)\nendif()\n'''
    patched = original.replace(anchor, normalization, 1)

    block_pattern = re.compile(
        r"if\(TARGET TCL::TCL\)\n"
        r"\s*# --- CONAN 2 LOGIC ---.*?"
        r"^endif\(\)\n",
        flags=re.MULTILINE | re.DOTALL,
    )
    replacement = '''if(TARGET tcl::tcl OR TARGET TCL::TCL)
    message(STATUS "Using imported Tcl target")
    if(DEFINED tcl_PACKAGE_FOLDER AND EXISTS "${tcl_PACKAGE_FOLDER}/lib/tcl8.6/init.tcl")
        set(TCL_INIT_FILE "${tcl_PACKAGE_FOLDER}/lib/tcl8.6/init.tcl")
    endif()
else()
    message(STATUS "Using System TCL")
    if(TCL_INCLUDE_PATH)
        target_include_directories(OpenSees PUBLIC ${TCL_INCLUDE_PATH})
    endif()
endif()

# Tcl runtime files are staged only when a valid location is known.  The Python
# build helper performs a second post-build search for distro/Conan layouts.
if(TCL_INIT_FILE AND EXISTS "${TCL_INIT_FILE}")
    get_filename_component(_VG_TCL_INIT_DIR "${TCL_INIT_FILE}" DIRECTORY)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib/tcl8.6")
    file(COPY "${_VG_TCL_INIT_DIR}/"
         DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/lib/tcl8.6"
         FILES_MATCHING PATTERN "*.tcl")
else()
    message(WARNING "Tcl init.tcl not located during configure; staging deferred to build helper")
endif()
'''
    patched, count = block_pattern.subn(replacement, patched, count=1)
    if count != 1:
        raise RuntimeError("OpenSees CMakeLists.txt Tcl executable block did not match v3.8.0")

    cmake_path.write_text(patched, encoding="utf-8")
    return {
        "applied": True,
        "already_present": False,
        "path": str(cmake_path),
        "original_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
        "sha256": sha256_file(cmake_path),
        "marker": _CMAKE_PATCH_MARKER,
        "scope": "CMake build integration only; no solver/model source modified",
    }


def _tcl_runtime_declared_version(runtime: Path) -> str | None:
    """Return the exact Tcl version required by a runtime's ``init.tcl``."""

    init_tcl = runtime / "init.tcl"
    if not init_tcl.is_file():
        return None
    content = init_tcl.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"package\s+require\s+-exact\s+Tcl\s+([0-9.]+)", content)
    return match.group(1) if match else None


def _tcl_runtime_candidates(search_roots: Sequence[Path]) -> list[Path]:
    """Discover Tcl runtimes in compatibility-first order.

    OpenSees must use the ``init.tcl`` matching the Tcl library it was linked
    against.  A Conan build therefore prefers the CMake-staged/Conan runtime;
    distribution runtimes are fallbacks only.  V0.17.2 did the reverse and
    could pair a Tcl 8.6.11 executable with Ubuntu's Tcl 8.6.12 ``init.tcl``.
    """

    ordered: list[Path] = []
    seen: set[str] = set()

    def add(runtime: Path) -> None:
        init_tcl = runtime / "init.tcl"
        if not init_tcl.is_file():
            return
        try:
            key = str(runtime.resolve())
        except OSError:
            key = str(runtime)
        if key not in seen:
            seen.add(key)
            ordered.append(runtime)

    for root in search_roots:
        if not root.exists():
            continue
        # Prefer the path staged by the selected CMake build before any broad
        # recursive match.  This is normally copied from the exact Conan Tcl.
        add(root / "lib" / "tcl8.6")
        add(root / "tcl8.6")
        try:
            matches = sorted(
                root.rglob("tcl8.6/init.tcl"),
                key=lambda value: (len(value.parts), str(value)),
            )
        except (OSError, PermissionError):
            matches = []
        for match in matches:
            add(match.parent)

    # System Tcl is deliberately last because its patch level can differ from
    # the library supplied by Conan even though both use the tcl8.6 directory.
    for init_tcl in (
        Path("/usr/share/tcltk/tcl8.6/init.tcl"),
        Path("/usr/lib/tcl8.6/init.tcl"),
        Path("/usr/lib/x86_64-linux-gnu/tcl8.6/init.tcl"),
    ):
        add(init_tcl.parent)
    return ordered


def _probe_tcl_runtime(
    *, binary: Path, runtime: Path, work_dir: Path, probe_index: int
) -> dict[str, object]:
    """Execute OpenSees with one candidate runtime and record compatibility."""

    work_dir.mkdir(parents=True, exist_ok=True)
    script = work_dir / f"probe-{probe_index:02d}.tcl"
    output = work_dir / f"probe-{probe_index:02d}.out"
    script.write_text(
        "\n".join(
            [
                "wipe",
                'puts stdout "VIBROGEMMA_TCL_PROBE=[info patchlevel]"',
                "exit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["TCL_LIBRARY"] = str(runtime)
    completed = subprocess.run(
        [str(binary), str(script)],
        cwd=work_dir,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    combined = completed.stdout + "\n--- STDERR ---\n" + completed.stderr
    output.write_text(combined, encoding="utf-8")
    patchlevel_match = re.search(r"VIBROGEMMA_TCL_PROBE=([0-9.]+)", completed.stdout)
    declared_version = _tcl_runtime_declared_version(runtime)
    interpreter_patchlevel = patchlevel_match.group(1) if patchlevel_match else None
    passed = (
        completed.returncode == 0
        and interpreter_patchlevel is not None
        and interpreter_patchlevel == declared_version
    )
    return {
        "runtime": str(runtime),
        "declared_version": declared_version,
        "interpreter_patchlevel": interpreter_patchlevel,
        "passed": passed,
        "returncode": completed.returncode,
        "log": str(output),
        "log_tail": _tail_text(output, 30),
    }


def _stage_tcl_runtime(
    *, cache_root: Path, source_root: Path, build_root: Path, binary: Path
) -> dict[str, object]:
    destination = cache_root / "lib" / "tcl8.6"
    candidates = _tcl_runtime_candidates(
        [build_root, source_root / "build", Path.home() / ".conan2" / "p"]
    )
    probes: list[dict[str, object]] = []
    selected: Path | None = None
    for index, runtime in enumerate(candidates):
        probe = _probe_tcl_runtime(
            binary=binary,
            runtime=runtime,
            work_dir=cache_root / "tcl-runtime-probes",
            probe_index=index,
        )
        probes.append(probe)
        if probe["passed"]:
            selected = runtime
            break
    if selected is None:
        summary = "\n\n".join(
            f"[{item['runtime']}]\n{item['log_tail']}" for item in probes
        )
        raise RuntimeError(
            "No Tcl 8.6 runtime was compatible with the built OpenSees executable. "
            f"Probes: {cache_root / 'tcl-runtime-probes'}\n{summary}"
        )
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(selected, destination)
    selected_probe = next(item for item in probes if item["passed"])
    declared_version = _tcl_runtime_declared_version(destination)
    return {
        "staged": True,
        "source": str(selected),
        "destination": str(destination),
        "declared_version": declared_version,
        "init_tcl_exact_version": declared_version,
        "interpreter_patchlevel": selected_probe.get("interpreter_patchlevel"),
        "init_tcl_sha256": sha256_file(destination / "init.tcl"),
        "candidate_count": len(candidates),
        "probes": probes,
    }


def smoke_test_binary(binary: Path, work_dir: Path) -> dict[str, object]:
    work_dir.mkdir(parents=True, exist_ok=True)
    script = work_dir / "smoke.tcl"
    output = work_dir / "smoke.out"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model BasicBuilder -ndm 1 -ndf 1",
                "node 1 0.0",
                "fix 1 1",
                'puts stdout "VIBROGEMMA_OPENSEES_SMOKE=OK"',
                "wipe",
                "exit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    tcl_runtime = binary.resolve().parent.parent / "lib" / "tcl8.6"
    if (tcl_runtime / "init.tcl").is_file():
        env["TCL_LIBRARY"] = str(tcl_runtime)
    completed = subprocess.run(
        [str(binary), str(script)],
        cwd=work_dir,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    output.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
    passed = completed.returncode == 0 and "VIBROGEMMA_OPENSEES_SMOKE=OK" in completed.stdout
    if not passed:
        raise RuntimeError(f"OpenSees smoke test failed; inspect {output}\n{_tail_text(output)}")
    return {
        "passed": True,
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "stdout_marker": True,
        "tcl_library": env.get("TCL_LIBRARY"),
        "log": str(output),
    }


def _configure_and_build(
    *,
    source_root: Path,
    build_root: Path,
    cache_root: Path,
    jobs: int,
    configure_args: Sequence[str],
    log_prefix: str,
    env: Mapping[str, str] | None = None,
    preserve_generated_files: bool = False,
) -> tuple[Path, dict[str, str]]:
    if build_root.exists() and not preserve_generated_files:
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True, exist_ok=True)
    configure_log = cache_root / f"cmake_configure_{log_prefix}.log"
    build_log = cache_root / f"cmake_build_{log_prefix}.log"
    _run_logged(
        [
            "cmake",
            "-S",
            str(source_root),
            "-B",
            str(build_root),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={cache_root / 'install'}",
            *[str(value) for value in configure_args],
        ],
        configure_log,
        cwd=source_root,
        env=env,
    )
    _run_logged(
        [
            "cmake",
            "--build",
            str(build_root),
            "--target",
            "OpenSees",
            "-j",
            str(max(1, jobs)),
        ],
        build_log,
        cwd=source_root,
        env=env,
    )
    return discover_binary(build_root), {
        "configure_log": str(configure_log),
        "build_log": str(build_log),
    }


def _system_package_versions() -> dict[str, str]:
    packages = ["libhdf5-dev", "tcl-dev", "libeigen3-dev", "zlib1g-dev", "liblapack-dev"]
    observed: dict[str, str] = {}
    dpkg = shutil.which("dpkg-query")
    if not dpkg:
        return observed
    for package in packages:
        completed = subprocess.run(
            [dpkg, "-W", "-f=${Version}", package],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            observed[package] = completed.stdout.strip()
    return observed


def recover_completed_build(
    *,
    source_archive: Path,
    cache_root: Path,
    expected_source_sha256: str | None = None,
    cmake_args: Sequence[str] = (),
) -> dict[str, object] | None:
    """Seal an already compiled OpenSees tree after a runtime-staging failure.

    This path performs no compilation.  It is intended for interrupted builds
    where CMake reached ``Built target OpenSees`` but the post-build Tcl smoke
    test failed.  The executable, source checksum, compatible runtime probes,
    and final smoke test are all recorded before a cache receipt is written.
    """

    source_archive = source_archive.resolve()
    cache_root = cache_root.resolve()
    observed_sha = sha256_file(source_archive)
    if expected_source_sha256 and observed_sha != expected_source_sha256:
        raise RuntimeError(
            f"OpenSees source SHA mismatch: expected={expected_source_sha256}, observed={observed_sha}"
        )

    extract_root = cache_root / "source"
    if not extract_root.is_dir():
        return None
    source_root = discover_source_root(extract_root)
    executable_candidates = [
        path
        for path in source_root.rglob("OpenSees")
        if path.is_file() and os.access(path, os.X_OK)
    ]
    cached_binary = cache_root / "bin" / "OpenSees"
    if cached_binary.is_file() and os.access(cached_binary, os.X_OK):
        built_binary = cached_binary
        selected_build_root = next(
            (path.parent for path in executable_candidates if path.name == "OpenSees"),
            source_root / "build" / "Release",
        )
    elif executable_candidates:
        executable_candidates.sort(key=lambda path: (len(path.parts), str(path)))
        original_binary = executable_candidates[0]
        selected_build_root = original_binary.parent
        cached_binary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_binary, cached_binary)
        cached_binary.chmod(0o755)
        built_binary = cached_binary
    else:
        return None

    cached_binary.parent.mkdir(parents=True, exist_ok=True)
    if built_binary != cached_binary:
        shutil.copy2(built_binary, cached_binary)
    cached_binary.chmod(0o755)

    tcl_runtime = _stage_tcl_runtime(
        cache_root=cache_root,
        source_root=source_root,
        build_root=selected_build_root,
        binary=cached_binary,
    )
    selected_logs = {
        "configure_log": str(cache_root / "cmake_configure_conan_recipe.log"),
        "build_log": str(cache_root / "cmake_build_conan_recipe.log"),
    }
    conan_executable = shutil.which("conan")
    conan_version = None
    if conan_executable:
        conan_version = subprocess.run(
            [conan_executable, "--version"],
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip() or None
    patch_fingerprint = hashlib.sha256(_CMAKE_PATCH_MARKER.encode("utf-8")).hexdigest()
    receipt: dict[str, object] = {
        "schema": _BUILD_SCHEMA,
        "source_archive": str(source_archive),
        "source_sha256": observed_sha,
        "source_root": str(source_root),
        "build_root": str(selected_build_root),
        "build_strategy": "conan_recipe_recovered",
        "binary": str(cached_binary),
        "binary_sha256": sha256_file(cached_binary),
        "conan_executable": conan_executable,
        "conan_version": conan_version,
        "system_package_versions": _system_package_versions(),
        "cmake_args": list(cmake_args),
        "cmake_patch": {
            "marker": _CMAKE_PATCH_MARKER,
            "path": str(source_root / "CMakeLists.txt"),
            "sha256": sha256_file(source_root / "CMakeLists.txt"),
            "scope": "CMake build integration only; no solver/model source modified",
        },
        "patch_fingerprint": patch_fingerprint,
        "attempts": [
            {
                "strategy": "conan_recipe",
                "passed": True,
                "recovered_from_completed_binary": True,
                **selected_logs,
            }
        ],
        "selected_logs": selected_logs,
        "tcl_runtime": tcl_runtime,
        "cache_reused": False,
        "recovered_without_recompile": True,
    }
    receipt["smoke_test"] = smoke_test_binary(cached_binary, cache_root / "smoke-v0173")
    (cache_root / "build_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def build_from_source(
    *,
    source_archive: Path,
    cache_root: Path,
    jobs: int,
    expected_source_sha256: str | None = None,
    cmake_args: Sequence[str] = (),
) -> dict[str, object]:
    """Build pinned OpenSees 3.8.0 with logged, restart-safe fallbacks.

    Strategy order:
      1. The v3.8.0 repository's Conan-2 recipe/workflow path.
      2. The repository's simpler ``conan.txt`` path.
      3. Ubuntu system development packages, still compiling the same pinned
         OpenSees source archive.

    The extracted CMakeLists receives a deterministic compatibility-only patch
    for inconsistent Tcl target spelling/runtime staging.  Every attempt and log
    is recorded; validation and model data are untouched.
    """

    source_archive = source_archive.resolve()
    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    observed_sha = sha256_file(source_archive)
    if expected_source_sha256 and observed_sha != expected_source_sha256:
        raise RuntimeError(
            f"OpenSees source SHA mismatch: expected={expected_source_sha256}, observed={observed_sha}"
        )

    receipt_path = cache_root / "build_receipt.json"
    cached_binary = cache_root / "bin" / "OpenSees"
    conan_executable = shutil.which("conan")
    conan_version = None
    if conan_executable:
        conan_version = subprocess.run(
            [conan_executable, "--version"],
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip() or None

    patch_fingerprint = hashlib.sha256(_CMAKE_PATCH_MARKER.encode("utf-8")).hexdigest()
    if receipt_path.is_file() and cached_binary.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema") == _BUILD_SCHEMA
            and receipt.get("source_sha256") == observed_sha
            and receipt.get("binary_sha256") == sha256_file(cached_binary)
            and receipt.get("cmake_args") == list(cmake_args)
            and receipt.get("patch_fingerprint") == patch_fingerprint
        ):
            receipt["cache_reused"] = True
            receipt["smoke_test"] = smoke_test_binary(cached_binary, cache_root / "smoke")
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return receipt

    extract_root = cache_root / "source"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True)
    shutil.unpack_archive(str(source_archive), str(extract_root))
    source_root = discover_source_root(extract_root)
    cmake_patch = _patch_opensees_380_cmake(source_root)

    attempts: list[dict[str, object]] = []
    built_binary: Path | None = None
    selected_strategy: str | None = None
    selected_logs: dict[str, str] = {}
    selected_build_root: Path | None = None

    if conan_executable:
        profile_log = cache_root / "conan_profile.log"
        try:
            _run_logged(
                [conan_executable, "profile", "detect", "--force"],
                profile_log,
                cwd=source_root,
            )
        except RuntimeError as exc:
            attempts.append({"strategy": "conan_profile", "passed": False, "error": str(exc)})
        else:
            for strategy in ("conan_recipe", "conan_txt"):
                try:
                    if strategy == "conan_recipe":
                        install_log = cache_root / "conan_install_recipe.log"
                        _run_logged(
                            [
                                conan_executable,
                                "install",
                                str(source_root),
                                "--build=missing",
                                "-s",
                                "build_type=Release",
                            ],
                            install_log,
                            cwd=source_root,
                        )
                        build_root = source_root / "build" / "Release"
                    else:
                        build_root = cache_root / "conan-txt-build"
                        if build_root.exists():
                            shutil.rmtree(build_root)
                        install_log = cache_root / "conan_install_txt.log"
                        _run_logged(
                            [
                                conan_executable,
                                "install",
                                str(source_root / "conan.txt"),
                                "--output-folder",
                                str(build_root),
                                "--build=missing",
                                "-s",
                                "build_type=Release",
                            ],
                            install_log,
                            cwd=source_root,
                        )
                    toolchains = [
                        build_root / "generators" / "conan_toolchain.cmake",
                        build_root / "conan_toolchain.cmake",
                    ]
                    toolchain = next((path for path in toolchains if path.is_file()), None)
                    if toolchain is None:
                        raise FileNotFoundError(
                            "Conan completed without a toolchain; checked: "
                            + ", ".join(str(path) for path in toolchains)
                        )
                    built_binary, selected_logs = _configure_and_build(
                        source_root=source_root,
                        build_root=build_root,
                        cache_root=cache_root,
                        jobs=jobs,
                        configure_args=[
                            f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
                            "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON",
                            "-DCONAN_EXPORTED=TRUE",
                            "-DCMAKE_DISABLE_FIND_PACKAGE_MPI=TRUE",
                            "-DCMAKE_DISABLE_FIND_PACKAGE_MKL=TRUE",
                            *list(cmake_args),
                        ],
                        log_prefix=strategy,
                        preserve_generated_files=True,
                    )
                    selected_strategy = strategy
                    selected_build_root = build_root
                    attempts.append(
                        {
                            "strategy": strategy,
                            "passed": True,
                            "install_log": str(install_log),
                            **selected_logs,
                        }
                    )
                    break
                except Exception as exc:  # continue to the next declared strategy
                    attempts.append(
                        {
                            "strategy": strategy,
                            "passed": False,
                            "error": str(exc),
                            "install_log": str(install_log) if "install_log" in locals() else None,
                        }
                    )
                    built_binary = None

    if built_binary is None:
        try:
            system_build_root = cache_root / "system-build"
            built_binary, selected_logs = _configure_and_build(
                source_root=source_root,
                build_root=system_build_root,
                cache_root=cache_root,
                jobs=jobs,
                configure_args=[
                    "-DNOT_USING_CONAN=TRUE",
                    "-DCMAKE_DISABLE_FIND_PACKAGE_MPI=TRUE",
                    "-DCMAKE_DISABLE_FIND_PACKAGE_MKL=TRUE",
                    "-DHDF5_PREFER_PARALLEL=FALSE",
                    *list(cmake_args),
                ],
                log_prefix="system_packages",
            )
            selected_strategy = "system_packages"
            selected_build_root = system_build_root
            attempts.append({"strategy": "system_packages", "passed": True, **selected_logs})
        except Exception as exc:
            attempts.append({"strategy": "system_packages", "passed": False, "error": str(exc)})
            combined = "\n\n".join(
                f"[{item.get('strategy')}] {item.get('error', 'failed')}" for item in attempts
                if not item.get("passed")
            )
            failure_receipt = {
                "schema": _BUILD_SCHEMA + "-failure",
                "source_archive": str(source_archive),
                "source_sha256": observed_sha,
                "source_root": str(source_root),
                "cmake_patch": cmake_patch,
                "patch_fingerprint": patch_fingerprint,
                "conan_version": conan_version,
                "attempts": attempts,
            }
            (cache_root / "build_failure_receipt.json").write_text(
                json.dumps(failure_receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                "All OpenSees 3.8.0 source-build strategies failed. "
                f"Failure receipt: {cache_root / 'build_failure_receipt.json'}\n{combined}"
            ) from exc

    if selected_build_root is None or selected_strategy is None:
        raise AssertionError("build strategy completed without selected build metadata")

    cached_binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_binary, cached_binary)
    cached_binary.chmod(0o755)
    tcl_runtime = _stage_tcl_runtime(
        cache_root=cache_root,
        source_root=source_root,
        build_root=selected_build_root,
        binary=cached_binary,
    )
    receipt = {
        "schema": _BUILD_SCHEMA,
        "source_archive": str(source_archive),
        "source_sha256": observed_sha,
        "source_root": str(source_root),
        "build_root": str(selected_build_root),
        "build_strategy": selected_strategy,
        "binary": str(cached_binary),
        "binary_sha256": sha256_file(cached_binary),
        "conan_executable": conan_executable,
        "conan_version": conan_version,
        "system_package_versions": _system_package_versions(),
        "cmake_args": list(cmake_args),
        "cmake_patch": cmake_patch,
        "patch_fingerprint": patch_fingerprint,
        "attempts": attempts,
        "selected_logs": selected_logs,
        "tcl_runtime": tcl_runtime,
        "cache_reused": False,
    }
    receipt["smoke_test"] = smoke_test_binary(cached_binary, cache_root / "smoke")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
