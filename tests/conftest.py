"""Every test gets its own CityPulse database, model file and salt, so tests never touch real data."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(autouse=True)
def isolated_citypulse(tmp_path, monkeypatch):
    monkeypatch.setenv("CITYPULSE_DB", str(tmp_path / "cp.db")); monkeypatch.setenv("CITYPULSE_MODEL", str(tmp_path / "risk.joblib"))
    monkeypatch.setenv("CITYPULSE_SALT", "test-salt"); monkeypatch.delenv("CITYPULSE_ADMIN_TOKEN", raising=False)
    from citypulse import ml; ml._cached[:] = [None, None]
    yield
