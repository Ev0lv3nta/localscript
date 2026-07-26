import json

from typer.testing import CliRunner

from app.cli.main import cli


runner = CliRunner()


def test_verify_rejects_invalid_lua_syntax():
    result = runner.invoke(cli, ["verify", "--code", "return function("])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "lua_syntax_error" in payload["errors"]


def test_verify_reports_degraded_mode_without_lua_runtime(monkeypatch):
    monkeypatch.setattr("app.validation.validators._find_lua_binary", lambda: None)
    monkeypatch.setattr("app.validation.validators._find_luac_binary", lambda: None)

    result = runner.invoke(cli, ["verify", "--code", "return 1"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["degraded_mode"] is True
    assert "lua_runtime_missing" in payload["degraded_codes"]
