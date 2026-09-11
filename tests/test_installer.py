"""Run: python3 -m unittest discover -s tests -v.

Tests execute temporary repository copies. Network clients, package installers,
compilers and privileged commands are doubles: this suite verifies orchestration
and recovery, not dependency installation or hardware compatibility.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pty
import re
import select
import shutil
import signal
import struct
import subprocess
import tempfile
import termios
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODES = {
    "codec-live": ("setup.sh", [], "live"),
    "codec-offline": ("setup.sh", ["--offline"], "offline"),
    "gemma-live": ("VibroAgent-Gemma/setup.sh", ["--live"], "live"),
    "gemma-offline": ("VibroAgent-Gemma/setup.sh", ["--offline"], "offline"),
    "gemma-demo": ("VibroAgent-Gemma/setup.sh", ["--demo"], "demo"),
}
PYTHON_DOUBLE = '''#!/usr/bin/python3
import pathlib, sys
venv=pathlib.Path(__file__).resolve().parents[1]
site=venv/'site-packages'; site.mkdir(exist_ok=True)
if len(sys.argv)>1 and sys.argv[1]=='--version': print('Python 3.12 (test double)')
elif 'import geniex' in sys.argv[-1]:
    lib=site/'geniex/lib'; (lib/'llama_cpp').mkdir(parents=True, exist_ok=True); print(lib)
else: print(site)
'''


def progress_cursor_positions(output, columns):
    """Track the cursor operations emitted by the real ASCII progress renderer."""
    csi = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")
    positions = {"ST ": [], "[1/2]": [], "[2/2]": []}
    column = offset = row_wraps = 0
    progress_row = False
    frame_wraps = []
    while offset < len(output):
        control = csi.match(output, offset)
        if control:
            params, command = control.groups()
            if command == "K":
                row_wraps = 0
                progress_row = False
            elif command == "A" and progress_row:
                frame_wraps.append(row_wraps)
            elif command == "G":
                column = int(params or "1") - 1
            elif command in {"H", "f"}:
                parts = params.split(";")
                column = int(parts[1] or "1") - 1 if len(parts) > 1 else 0
            offset = control.end()
            continue
        character = output[offset]
        if character == "\r":
            column = 0
        elif character != "\n" and ord(character) >= 32:
            if column >= columns:
                column = 0
                row_wraps += 1
            for marker, seen in positions.items():
                if output.startswith(marker, offset):
                    seen.append(column)
                    if marker == "ST ":
                        progress_row = True
            column += 1
        offset += 1
    return positions, frame_wraps


class InstallerFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="vibro-installer-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.repo = self.base / "user's repo with spaces"
        self.repo.mkdir()
        for source in list(ROOT.glob("*.sh")) + list((ROOT / "VibroAgent-Gemma").glob("*.sh")):
            dest = self.repo / source.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
        for rel in ["models/release_assets_manifest.json", "VibroAgent-Gemma/models/gemma_release_assets_manifest.json"]:
            dest = self.repo / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dest)
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.trace = self.base / "trace"
        self.home = self.base / "home"
        self.home.mkdir()
        # Only the disposable child process gets a temporary HOME.
        self.env = dict(os.environ, PATH=f"{self.bin}:/usr/bin:/bin", HOME=str(self.home),
                        TRACE=str(self.trace), FIXTURE_ROOT=str(self.repo),
                        TERM="xterm", LANG="C.UTF-8", NO_COLOR="1")
        for key in list(self.env):
            if key.startswith(("VIBRO", "GENIEX", "HF_", "GITHUB_", "GH_", "UV_")):
                self.env.pop(key)
        self.env.update(VIBRO_CODEC_VENV=str(self.home / "codec-cpu-venv"),
                        GENIEX_VENV=str(self.home / "geniex-venv"))
        for command in ["curl", "gh", "git", "sudo", "apt-get", "usermod", "groupadd", "udevadm", "g++"]:
            self.stub(command, 'echo "unexpected external command: $0 $*" >&2\nexit 97\n')
        self.stub("id", 'if [ "$1" = -u ]; then echo 0; else /usr/bin/id "$@"; fi\n')
        (self.base / "python-double").write_text(PYTHON_DOUBLE)
        self.env["PYTHON_DOUBLE"] = str(self.base / "python-double")
        self.stub("uv", '''python3 - "$@" <<'PY'
import json, os, pathlib, sys
args=sys.argv[1:]
with open(os.environ['TRACE'], 'a') as f: f.write(json.dumps(['uv']+args)+'\\n')
if os.environ.get('FAIL_UV'): sys.exit(42)
if args[0] == 'venv':
    venv=pathlib.Path(args[-1]); (venv/'bin').mkdir(parents=True, exist_ok=True)
    (venv/'bin/python').write_text(pathlib.Path(os.environ['PYTHON_DOUBLE']).read_text())
    (venv/'bin/python').chmod(0o755)
    (venv/'bin/geniex-py').write_text('#!/bin/sh\\necho "GenieX test double"\\n')
    (venv/'bin/geniex-py').chmod(0o755)
PY
''')

    def stub(self, name, body, repo=False):
        path = (self.repo if repo else self.bin) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
        path.chmod(0o755)
        return path

    def run_script(self, script, *args, env=None):
        return subprocess.run([str(self.repo / script), *args], cwd=self.base,
                              env=self.env | (env or {}), stdin=subprocess.DEVNULL,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)

    def calls(self):
        return self.trace.read_text().splitlines() if self.trace.exists() else []

    def prepare_setup(self):
        self.stub("setup_usb.sh", 'echo usb >> "$TRACE"\n', repo=True)
        self.stub("setup_sdk.sh", '''echo sdk >> "$TRACE"
mkdir -p "$FIXTURE_ROOT/stdatalog_core"
printf 'install_requires=[\n"numpy",\n"stdatalog_pnpl",\n]\n' > "$FIXTURE_ROOT/stdatalog_core/setup.py"
''', repo=True)
        for name in ["setup_models.sh", "setup_recordings.sh", "VibroAgent-Gemma/setup_models.sh"]:
            self.stub(name, 'echo "$(basename "$0")" >> "$TRACE"\n', repo=True)
        self.stub("git", 'echo git >> "$TRACE"\n')
        self.stub("g++", '''echo compile >> "$TRACE"
while [ "$#" -gt 0 ]; do
 if [ "$1" = -o ]; then shift; mkdir -p "$(dirname "$1")"; touch "$1"; fi
 shift
done
''')

    def run_tty(self, *args, keys=None, signal_on=None, rows=24, cols=80, translate_newlines=True):
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        attributes = termios.tcgetattr(slave)
        attributes[1] |= termios.OPOST
        if translate_newlines:
            attributes[1] |= termios.ONLCR
        else:
            attributes[1] &= ~termios.ONLCR
        termios.tcsetattr(slave, termios.TCSANOW, attributes)
        process = subprocess.Popen([str(self.repo / "install.sh"), *args], cwd=self.base,
                                   env=self.env, stdin=slave, stdout=slave, stderr=slave,
                                   start_new_session=True)
        os.close(slave)
        output = bytearray()
        sent = False
        deadline = time.monotonic() + 12
        try:
            while time.monotonic() < deadline:
                if select.select([master], [], [], 0.05)[0]:
                    try:
                        data = os.read(master, 65536)
                    except OSError:
                        break
                    if not data:
                        break
                    output.extend(data)
                if not sent and keys is not None and b"q quit" in output:
                    os.write(master, keys); sent = True
                if not sent and signal_on and signal_on.encode() in output:
                    os.kill(process.pid, signal.SIGTERM); sent = True
            try:
                code = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.fail("installer hung: " + output.decode(errors="replace")[-1200:])
            return code, output.decode(errors="replace")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
            os.close(master)


class EntrypointTests(InstallerFixture):
    def test_shell_syntax_and_executable_bits(self):
        for script in self.repo.rglob("*.sh"):
            with self.subTest(script=script.name):
                self.assertTrue(os.access(script, os.X_OK))
                self.assertEqual(subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode, 0)

    def test_all_dry_run_mappings_and_no_writes(self):
        for mode, (script, args, _) in MODES.items():
            with self.subTest(mode=mode):
                result = self.run_script("install.sh", "--" + mode, "--dry-run")
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn(script, result.stdout)
                self.assertIn(" ".join(args), result.stdout)
                self.assertNotIn("\x1b", result.stdout)
                self.assertFalse((self.repo / ".run").exists())
                self.assertEqual(self.calls(), [])

    def test_invalid_arguments_and_nonterminal_selection(self):
        for args in [[], ["--wrong"], ["--codec-live", "--gemma-demo"], ["--gemma-demo", "--gemma-demo"], ["--preview"]]:
            with self.subTest(args=args):
                result = self.run_script("install.sh", *args)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertEqual(self.calls(), [])

    def test_help_does_not_install(self):
        result = self.run_script("install.sh", "--help")
        self.assertEqual(result.returncode, 0)
        for mode in MODES:
            self.assertIn("--" + mode, result.stdout)
        self.assertEqual(self.calls(), [])

    def test_dispatch_and_logging_for_each_mode(self):
        for mode, (script, args, _) in MODES.items():
            with self.subTest(mode=mode):
                self.stub(script, 'printf "called:<%s>\\n" "$@"\n', repo=True)
                result = self.run_script("install.sh", "--" + mode, "--no-animation")
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn(f"called:<{args[0] if args else ''}>", result.stdout)
                self.assertIn("called:", (self.repo / ".run/install.log").read_text())
                self.assertIn("Next", result.stdout)

    def test_live_install_reuses_noninteractive_sudo(self):
        self.stub("id", "echo 1000\n")
        self.stub("sudo", 'if [ "$*" = "-n true" ]; then exit 0; fi\nexit 42\n')
        self.stub("setup.sh", "echo authorized-live-setup\n", repo=True)
        result = self.run_script("install.sh", "--codec-live")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("authorized-live-setup", result.stdout)

    def test_live_authorization_failure_stops_before_setup(self):
        self.stub("id", "echo 1000\n")
        self.stub("sudo", 'if [ "$*" = "-n true" ]; then exit 1; fi\nexit 42\n')
        self.stub("setup.sh", 'echo ran >> "$TRACE"\n', repo=True)
        result = self.run_script("install.sh", "--codec-live")
        self.assertEqual(result.returncode, 42, result.stdout)
        self.assertEqual(self.calls(), [])

    def test_failure_status_and_log_are_preserved(self):
        self.stub("setup.sh", 'echo fixture-failure\nexit 42\n', repo=True)
        result = self.run_script("install.sh", "--codec-offline")
        self.assertEqual(result.returncode, 42, result.stdout)
        self.assertIn("fixture-failure", (self.repo / ".run/install.log").read_text())
        self.assertNotIn("Next", result.stdout)

    def test_logging_failure_stops_before_setup(self):
        (self.repo / ".run").write_text("not a directory")
        self.stub("setup.sh", 'echo ran >> "$TRACE"\n', repo=True)
        result = self.run_script("install.sh", "--codec-offline")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.calls(), [])

    def test_tee_failure_is_reported(self):
        self.stub("setup.sh", "echo fixture-output\n", repo=True)
        self.stub("tee", "cat >/dev/null\nexit 74\n")
        result = self.run_script("install.sh", "--codec-offline")
        self.assertEqual(result.returncode, 74, result.stdout)
        self.assertNotIn("Next", result.stdout)

    def test_menu_number_and_arrow_navigation(self):
        for keys, label in [(b"2\r", "Codec / offline replay"), (b"\x1b[B\r", "Codec / offline replay"), (b"5\r", "Gemma / direct LUMO demo")]:
            with self.subTest(keys=keys):
                code, output = self.run_tty("--no-splash", "--dry-run", "--ascii", keys=keys)
                self.assertEqual(code, 0, output[-1000:])
                self.assertIn(label, output)
                self.assertIn("\x1b[?1049l", output)
                self.assertIn("\x1b[?25h", output)

    def test_preview_preserves_previous_install_log(self):
        log = self.repo / ".run/install.log"
        log.parent.mkdir()
        log.write_text("previous installation evidence")
        self.stub("sleep", "exit 0\n")
        code, output = self.run_tty("--preview-animation", "--no-animation")
        self.assertEqual(code, 0, output[-1200:])
        self.assertEqual(log.read_text(), "previous installation evidence")
        code, output = self.run_tty("--preview-animation", "--dry-run")
        self.assertEqual(code, 0, output[-1200:])
        self.assertIn("dry run", output)
        self.assertEqual(log.read_text(), "previous installation evidence")

    def test_menu_quit(self):
        code, output = self.run_tty("--no-splash", keys=b"q", rows=40, cols=120)
        self.assertEqual(code, 0, output[-1000:])
        self.assertIn("Installer closed", output)
        self.assertFalse((self.repo / ".run").exists())

    def test_cancellation_stops_setup_descendants(self):
        self.stub("setup.sh", '''sleep 30 &
echo $! > "$FIXTURE_ROOT/worker.pid"
echo "==== CANCELLATION_READY ===="
wait
''', repo=True)
        code, output = self.run_tty("--codec-offline", "--ascii", signal_on="CANCELLATION_READY")
        pid = int((self.repo / "worker.pid").read_text())
        try:
            self.assertEqual(code, 130, output[-1200:])
            time.sleep(0.1)
            stat = Path(f"/proc/{pid}/stat")
            if stat.exists():
                self.assertEqual(stat.read_text().split()[2], "Z", "setup child survived cancellation")
        finally:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_plain_output_cancellation_is_prompt(self):
        self.stub("setup.sh", '''sleep 30 &
echo $! > "$FIXTURE_ROOT/worker.pid"
echo "CANCELLATION_READY"
wait
''', repo=True)
        code, output = self.run_tty("--codec-offline", "--no-animation", signal_on="CANCELLATION_READY")
        self.assertEqual(code, 130, output[-1200:])

    def test_animated_success_and_failure(self):
        for status in [0, 42]:
            with self.subTest(status=status):
                self.stub("setup.sh", f'echo "==== [1/2] Fixture ==== "\necho progress\nsleep 0.2\necho final-message\nexit {status}\n', repo=True)
                code, output = self.run_tty("--codec-offline", "--ascii", cols=48)
                self.assertEqual(code, status, output[-1200:])
                self.assertIn("\x1b[?25h", output)
                self.assertIn("final-message", (self.repo / ".run/install.log").read_text())
                self.assertIn("Installed" if status == 0 else "Installation failed", output)

    def test_animated_rows_keep_their_column_and_do_not_wrap(self):
        self.stub("setup.sh", '''echo "==== [1/2] First phase with a long title ===="
printf '%120s\\n' 'long detail'
sleep 0.4
echo "==== [2/2] Second phase with a long title ===="
printf '%120s\\n' 'another detail'
sleep 0.4
''', repo=True)
        for cols, translate_newlines in [(120, True), (120, False), (40, True), (40, False), (24, False)]:
            with self.subTest(cols=cols, translate_newlines=translate_newlines):
                code, output = self.run_tty(
                    "--codec-offline", "--ascii", cols=cols,
                    translate_newlines=translate_newlines,
                )
                self.assertEqual(code, 0, output[-1200:])
                positions, frame_wraps = progress_cursor_positions(output, cols)
                indent = max(0, (cols - 78) // 2 - 2)
                self.assertEqual(set(positions["ST "]), {indent + 2})
                self.assertEqual(set(positions["[1/2]"]), {indent + 4})
                self.assertEqual(set(positions["[2/2]"]), {indent + 4})
                self.assertGreaterEqual(len(frame_wraps), 2)
                self.assertEqual(set(frame_wraps), {0}, "animated rows must fit without automatic wrapping")

    def test_tiny_terminal_uses_plain_output_without_cursor_redraw(self):
        self.stub("setup.sh", "echo tiny-output\nexit 42\n", repo=True)
        code, output = self.run_tty("--codec-offline", "--ascii", cols=20)
        self.assertEqual(code, 42)
        self.assertIn("tiny-output", output)
        self.assertNotIn("\x1b[1A", output)


class SetupTests(InstallerFixture):
    def setUp(self):
        super().setUp()
        self.prepare_setup()

    def test_every_mode_and_virtual_environment_reuse(self):
        for mode, (script, args, expected) in MODES.items():
            with self.subTest(mode=mode):
                self.trace.unlink(missing_ok=True)
                result = self.run_script(script, *args)
                self.assertEqual(result.returncode, 0, result.stdout)
                marker = self.repo / Path(script).parent / ".vibro_mode"
                self.assertEqual(marker.read_text().strip(), expected)
                self.assertEqual("usb" in self.calls(), expected == "live")
                self.assertEqual("sdk" in self.calls(), expected != "demo")
                self.assertEqual("setup_recordings.sh" in self.calls(), mode == "codec-offline")
                self.trace.unlink()
                again = self.run_script(script, *args)
                self.assertEqual(again.returncode, 0, again.stdout)
                self.assertFalse(any('"venv"' in call for call in self.calls()), self.calls())

    def test_failure_preserves_previous_mode(self):
        for script, args, _ in MODES.values():
            with self.subTest(script=script, args=args):
                marker = self.repo / Path(script).parent / ".vibro_mode"
                marker.write_text("previous\n")
                result = self.run_script(script, *args, env={"FAIL_UV": "1"})
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertEqual(marker.read_text(), "previous\n")

    def test_fresh_failure_does_not_publish_mode(self):
        for script in ["setup.sh", "VibroAgent-Gemma/setup.sh"]:
            with self.subTest(script=script):
                result = self.run_script(script, "--offline", env={"FAIL_UV": "1"})
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse((self.repo / Path(script).parent / ".vibro_mode").exists())

    def test_conflicting_gemma_modes_rejected_without_work(self):
        result = self.run_script("VibroAgent-Gemma/setup.sh", "--live", "--demo")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertEqual(self.calls(), [])

    def test_sdk_pythonpath_file_preserves_space_containing_paths(self):
        for script in ["setup.sh", "VibroAgent-Gemma/setup.sh"]:
            with self.subTest(script=script):
                result = self.run_script(script, "--offline")
                self.assertEqual(result.returncode, 0, result.stdout)
        files = list(self.repo.rglob("vibroagent_sdk.pth"))
        self.assertEqual(len(files), 2)
        expected = [str(self.repo / name) for name in ["stdatalog_core", "stdatalog_pnpl", "stdatalog_dtk", "stdatalog_gui"]]
        for path in files:
            self.assertEqual(path.read_text().splitlines(), expected)

    def test_uv_is_bootstrapped_when_missing(self):
        saved = self.base / "saved-uv"
        (self.bin / "uv").rename(saved)
        self.env["SAVED_UV"] = str(saved)
        self.stub("curl", '''cat <<'BOOTSTRAP'
mkdir -p "$HOME/.local/bin"
cp "$SAVED_UV" "$HOME/.local/bin/uv"
chmod +x "$HOME/.local/bin/uv"
BOOTSTRAP
''')
        result = self.run_script("setup.sh", "--offline")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertTrue((self.home / ".local/bin/uv").is_file())
        self.assertTrue((self.repo / "vibroagent-venv/bin/python").is_file())

    def test_bootstrap_failure_does_not_publish_mode(self):
        (self.bin / "uv").unlink()
        self.stub("curl", "exit 22\n")
        result = self.run_script("setup.sh", "--offline")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse((self.repo / ".vibro_mode").exists())
        self.assertEqual(self.calls(), [])

    def test_late_recordings_failure_preserves_mode(self):
        marker = self.repo / ".vibro_mode"
        marker.write_text("live\n")
        self.stub("setup_recordings.sh", "exit 42\n", repo=True)
        result = self.run_script("setup.sh", "--offline")
        self.assertEqual(result.returncode, 42, result.stdout)
        self.assertEqual(marker.read_text(), "live\n")

    def test_gemma_environment_is_isolated(self):
        result = self.run_script("VibroAgent-Gemma/setup.sh", "--demo")
        self.assertEqual(result.returncode, 0, result.stdout)
        venv_calls = [json.loads(call) for call in self.calls() if call.startswith('["uv", "venv"')]
        self.assertEqual(len(venv_calls), 1)
        self.assertNotIn("--system-site-packages", venv_calls[0])


class LauncherTests(InstallerFixture):
    def test_required_logger_failure_propagates(self):
        path = self.repo / "VibroAgent-Gemma/vibroagent.sh"
        prefix = path.read_text().split("# ---------------------------------------------------------------------------- main")[0]
        path.write_text(prefix + """
start_geniex() { return 0; }
start_webchat() { return 0; }
start_logger() { return 42; }
do_status() { return 0; }
do_start
""")
        result = self.run_script("VibroAgent-Gemma/vibroagent.sh")
        self.assertEqual(result.returncode, 42, result.stdout)

    def test_demo_uses_selected_model_endpoint_and_bundle(self):
        path = self.repo / "VibroAgent-Gemma/vibroagent.sh"
        prefix, main = path.read_text().split("# ---------------------------------------------------------------------------- main", 1)
        path.write_text(prefix + "\nstart_geniex() { return 0; }\n" + main)
        python = self.stub("demo-python", 'printf "<%s>\\n" "$@"\n')
        result = self.run_script("VibroAgent-Gemma/vibroagent.sh", "demo", "target_5", env={
            "GENIEX_PORT": "28281", "VIBROGEMMA_PYTHON": str(python),
            "VIBROGEMMA_BUNDLE": str(self.base / "custom bundle")})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("<http://127.0.0.1:28281/v1>", result.stdout)
        self.assertIn(f"<{self.base / 'custom bundle'}>", result.stdout)
        self.assertIn("<target_5>", result.stdout)


class PatchSeriesTests(InstallerFixture):
    def setUp(self):
        super().setUp()
        self.prepare_setup()
        self.geniex = self.repo / "VibroAgent-Gemma/vibrodiag_mcp_prototype/.run/GenieX-v0.4.0"
        self.geniex.mkdir(parents=True)
        self.patchdir = self.repo / "VibroAgent-Gemma/vibrodiag_mcp_prototype/patches"
        self.patchdir.mkdir()
        self.git("init", "-q")
        self.source = self.geniex / "source.txt"
        self.source.write_text("header\nbase\ntail\n")
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-qm", "base")
        self.source.write_text("header\nfirst patch\ntail\n")
        self.first = self.patchdir / "0001-feat-add-stateful-external-embedding-prefill.patch"
        self.first.write_text(self.git("diff").stdout)
        self.git("add", ".")
        self.source.write_text("header\nsecond patch\ntail\n")
        self.second = self.patchdir / "0002-feat-read-state-logits.patch"
        self.second.write_text(self.git("diff").stdout)
        self.git("add", ".")
        self.source.write_text("header\nthird patch\ntail\n")
        self.third = self.patchdir / "0003-fix-logit-validation-and-stop-sequences.patch"
        self.third.write_text(self.git("diff").stdout)
        # This reset is confined to the synthetic repository created above.
        self.git("reset", "--hard", "HEAD")
        self.stub("git", 'if [ "${3:-}" = submodule ]; then exit 0; fi\nexec /usr/bin/git "$@"\n')

    def git(self, *args):
        return subprocess.run(["/usr/bin/git", "-C", str(self.geniex), *args],
                              env=self.env, capture_output=True, text=True, check=True)

    def test_clean_and_repeated_overlapping_patch_series(self):
        for attempt in range(2):
            result = self.run_script("VibroAgent-Gemma/setup_geniex.sh")
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(self.source.read_text(), "header\nthird patch\ntail\n")
            if attempt:
                self.assertIn("patch series already applied", result.stdout)
        self.assertEqual(list(self.geniex.parent.glob(".patch-check.*")), [])

    def test_partial_patch_series_is_completed(self):
        self.git("apply", str(self.first))
        result = self.run_script("VibroAgent-Gemma/setup_geniex.sh")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.source.read_text(), "header\nthird patch\ntail\n")

    def test_unexpected_source_edit_is_preserved_and_rejected(self):
        self.source.write_text("unrelated local edit\n")
        result = self.run_script("VibroAgent-Gemma/setup_geniex.sh")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.source.read_text(), "unrelated local edit\n")

    def test_existing_two_patch_installation_receives_third_patch(self):
        self.git("apply", str(self.first), str(self.second))
        result = self.run_script("VibroAgent-Gemma/setup_geniex.sh")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.source.read_text(), "header\nthird patch\ntail\n")


class SdkTests(InstallerFixture):
    def test_failed_sdk_fetch_preserves_existing_tree(self):
        sdk = self.repo / "stdatalog_core"
        sdk.mkdir()
        (sdk / "previous.txt").write_text("existing SDK")
        self.stub("git", "exit 42\n")
        result = self.run_script("setup_sdk.sh")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual((sdk / "previous.txt").read_text(), "existing SDK")
        self.assertEqual(list(self.repo.glob(".stdatalog_core.*")), [])

    def test_usb_library_check_drains_ldconfig_under_pipefail(self):
        script = self.repo / "setup_usb.sh"
        rules = self.base / "rules"
        rules.mkdir()
        script.write_text(script.read_text().replace("/etc/udev/rules.d", str(rules)))
        self.stub("ldconfig", '''python3 - <<'LIBS'
import os
os.write(1, b'libusb-1.0.so.0 => /lib/libusb-1.0.so.0\\n')
for _ in range(100): os.write(1, b'x' * 8192 + b'\\n')
LIBS
''')
        self.stub("id", 'if [ "$1" = -u ]; then echo 0; else echo "ubuntu hsdatalog"; fi\n')
        self.stub("getent", "echo hsdatalog:x:1001:ubuntu\n")
        self.stub("udevadm", "exit 0\n")
        result = self.run_script("setup_usb.sh")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("already present", result.stdout)


class DownloadTests(InstallerFixture):
    def model_fixture(self):
        name = "qwen3_4b_codes_v3_Q4_0_embq8.gguf"
        payload = b"fixture model bytes"
        assets = self.base / "assets"
        assets.mkdir()
        parts = []
        for index, data in enumerate([payload[:7], payload[7:]]):
            part = f"{name}.part-{index:02d}.part"
            (assets / part).write_bytes(data)
            parts.append({"name": part, "sha256": hashlib.sha256(data).hexdigest()})
        manifest = {"gguf": {"parts": parts}}
        encoded = json.dumps(manifest)
        (assets / "release_assets_manifest.json").write_text(encoded)
        (self.repo / "models/release_assets_manifest.json").write_text(encoded)
        script = self.repo / "setup_models.sh"
        script.write_text(script.read_text().replace(
            "3a18c057e47d8032cb771140e54ed7bbfcf8cf1d58c6d990f579800f149a90c2",
            hashlib.sha256(payload).hexdigest()))
        self.env["ASSETS"] = str(assets)
        self.stub("gh", r'''asset=""; dest=""
while [ "$#" -gt 0 ]; do
 case "$1" in -p) shift; asset="$1" ;; -O) shift; dest="$1" ;; esac
 shift
done
cp "$ASSETS/$asset" "$dest"
''')
        return name, payload, assets

    def test_model_parts_verified_and_reused_in_space_containing_path(self):
        name, payload, _ = self.model_fixture()
        stale = self.repo / "models/.parts"
        stale.mkdir()
        (stale / "stale.part").write_bytes(b"unrelated old partial download")
        result = self.run_script("setup_models.sh")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual((self.repo / "models" / name).read_bytes(), payload)
        self.stub("gh", "exit 97\n")
        again = self.run_script("setup_models.sh")
        self.assertEqual(again.returncode, 0, again.stdout)
        self.assertIn("already in place", again.stdout)
        self.assertEqual(list((self.repo / "models").glob(".download.*")), [])

    def test_corrupt_model_part_falls_back_and_preserves_old_target(self):
        name, _, assets = self.model_fixture()
        (assets / (name + ".part-00.part")).write_bytes(b"corrupt")
        target = self.repo / "models" / name
        target.write_bytes(b"previous model")
        result = self.run_script("setup_models.sh")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("falling back to Hugging Face", result.stdout)
        self.assertEqual(target.read_bytes(), b"previous model")
        self.assertEqual(list(target.parent.glob(".download.*")), [])

    def test_valid_recordings_reused_but_changed_manifest_rejected(self):
        target = self.repo / "recordings"
        (target / "live_baseline").mkdir(parents=True)
        payload = b"fixture recording"
        (target / "live_baseline/data.dat").write_bytes(payload)
        manifest = {"slots": {"live_baseline": {"seconds": 300,
                    "files": {"data.dat": hashlib.sha256(payload).hexdigest()}}}}
        encoded = json.dumps(manifest).encode()
        (target / "recordings_manifest.json").write_bytes(encoded)
        release = {"recordings": {"manifest_sha256": hashlib.sha256(encoded).hexdigest(),
                                 "sha256": "0" * 64}}
        (self.repo / "models/release_assets_manifest.json").write_text(json.dumps(release))
        result = self.run_script("setup_recordings.sh")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("already in place", result.stdout)
        (target / "recordings_manifest.json").write_text('{"slots": {}}')
        result = self.run_script("setup_recordings.sh")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual((target / "live_baseline/data.dat").read_bytes(), payload)

    def test_gemma_split_reassembly_preserves_bytes_and_cleans_parts(self):
        name = "gemma-4-e2b-g1-Q8_0.gguf"
        payload = b"verified Gemma fixture model"
        assets = self.base / "gemma-assets"
        assets.mkdir()
        parts = []
        for index, data in enumerate([payload[:8], payload[8:16], payload[16:]]):
            part = f"{name}.{index}.part"
            (assets / part).write_bytes(data)
            parts.append({"name": part, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        manifest = json.dumps({"gguf": {"parts": parts}})
        (assets / "gemma_release_assets_manifest.json").write_text(manifest)
        (self.repo / "VibroAgent-Gemma/models/gemma_release_assets_manifest.json").write_text(manifest)
        script = self.repo / "VibroAgent-Gemma/setup_models.sh"
        script.write_text(script.read_text().replace(
            "11ceefee8d62080072fe2b65f68beab5e0f31ad716c640178ed93b5ac0eb31d6",
            hashlib.sha256(payload).hexdigest()))
        self.env["ASSETS"] = str(assets)
        self.stub("gh", '''while [ "$1" != -D ]; do shift; done
shift; cp "$ASSETS/"* "$1/"
''')
        result = self.run_script("VibroAgent-Gemma/setup_models.sh")
        self.assertEqual(result.returncode, 0, result.stdout)
        bundle = self.repo / "VibroAgent-Gemma/models/vibroagent-gemma-g1"
        self.assertEqual((bundle / name).read_bytes(), payload)
        self.assertEqual(list(bundle.glob(".download.*")), [])
        self.stub("gh", "exit 97\n")
        result = self.run_script("VibroAgent-Gemma/setup_models.sh")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("already present", result.stdout)

    def test_corrupt_gemma_local_model_preserves_target(self):
        bundle = self.repo / "VibroAgent-Gemma/models/vibroagent-gemma-g1"
        bundle.mkdir()
        target = bundle / "gemma-4-e2b-g1-Q8_0.gguf"
        target.write_bytes(b"old model")
        source = self.base / "bad.gguf"
        source.write_bytes(b"corrupt download")
        result = self.run_script("VibroAgent-Gemma/setup_models.sh", env={"VIBROGEMMA_GGUF_SOURCE": str(source)})
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(target.read_bytes(), b"old model")
        self.assertEqual(list(bundle.glob(".download.*")), [])

    def test_empty_recordings_manifest_cannot_pass(self):
        target = self.repo / "recordings"
        target.mkdir()
        (target / "recordings_manifest.json").write_text('{"slots": {}}')
        result = self.run_script("setup_recordings.sh")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("already in place", result.stdout)

    def test_corrupt_recordings_archive_falls_back(self):
        self.stub("gh", '''while [ "$1" != -O ]; do shift; done
shift; echo 'broken archive' > "$1"
''')
        result = self.run_script("setup_recordings.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("falling back to Hugging Face", result.stdout)
        self.assertTrue(any('"tool", "run"' in call for call in self.calls()), result.stdout)


if __name__ == "__main__":
    unittest.main()
