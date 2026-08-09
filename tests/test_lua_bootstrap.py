import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LUA = PROJECT_ROOT / ".tools/lua54/bin/lua"
LUAC = PROJECT_ROOT / ".tools/lua54/bin/luac"


def test_bootstrapped_lua_executes_real_unit_program():
    completed = subprocess.run(
        [
            str(LUA),
            "-e",
            "local values={2,3,5}; assert(#values==3); assert(values[1]+values[3]==7)",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0


def test_bootstrapped_luac_rejects_invalid_program():
    completed = subprocess.run(
        [str(LUAC), "-p", "-"],
        cwd=PROJECT_ROOT,
        input="return function(",
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "stdin:1:" in completed.stderr
