from app.core import config as config_module
from app.core.runtime_lock import build_runtime_lock, load_runtime_lock, write_runtime_lock


def test_runtime_lock_roundtrip_opt_in(monkeypatch, tmp_path):
    lock_path = tmp_path / ".runtime_profile.lock.json"
    monkeypatch.setenv("LOCALSCRIPT_RUNTIME_LOCK_PATH", str(lock_path))
    monkeypatch.setenv("LOCALSCRIPT_USE_RUNTIME_LOCK", "1")
    config_module.get_runtime_profile.cache_clear()
    profile = config_module.get_runtime_profile()

    payload = build_runtime_lock(
        profile=profile,
        selected_model=profile.fallback_model,
        selection_reason="test_fallback",
        quality_report={"ok": True},
        vram_report={"status": "skipped"},
    )
    write_runtime_lock(payload)
    loaded = load_runtime_lock()

    assert loaded["selected_model"] == profile.fallback_model
    config_module.get_runtime_profile.cache_clear()
    locked_profile = config_module.get_runtime_profile()
    assert locked_profile.model == profile.fallback_model


def test_runtime_lock_ignored_by_default(monkeypatch, tmp_path):
    lock_path = tmp_path / ".runtime_profile.lock.json"
    monkeypatch.setenv("LOCALSCRIPT_RUNTIME_LOCK_PATH", str(lock_path))
    monkeypatch.delenv("LOCALSCRIPT_USE_RUNTIME_LOCK", raising=False)
    config_module.get_runtime_profile.cache_clear()
    profile = config_module.get_runtime_profile()

    payload = build_runtime_lock(
        profile=profile,
        selected_model=profile.fallback_model,
        selection_reason="test_fallback",
        quality_report={"ok": True},
        vram_report={"status": "skipped"},
    )
    write_runtime_lock(payload)
    config_module.get_runtime_profile.cache_clear()
    unlocked_profile = config_module.get_runtime_profile()
    assert unlocked_profile.model == profile.model
