from pathlib import Path
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


@pytest.fixture(autouse=True)
def _allow_tmp_sensor_config_dirs(monkeypatch):
    """Tests write sensor registries under the pytest tmp tree. Production trusts only the
    project tree unless VIBRO_ALLOWED_SENSOR_CONFIG_DIR names more directories."""
    monkeypatch.setenv("VIBRO_ALLOWED_SENSOR_CONFIG_DIR", tempfile.gettempdir())
