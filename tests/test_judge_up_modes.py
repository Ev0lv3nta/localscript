import os
import stat
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def _write_executable(path, content):
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _prepare_fake_project(tmp_path):
    root = tmp_path / "project"
    scripts_dir = root / "scripts"
    venv_bin = root / ".venv" / "bin"
    fake_bin = root / "fake-bin"
    scripts_dir.mkdir(parents=True)
    venv_bin.mkdir(parents=True)
    fake_bin.mkdir(parents=True)

    source_script = Path(__file__).resolve().parents[1] / "scripts" / "judge_up.sh"
    (scripts_dir / "judge_up.sh").write_text(source_script.read_text(encoding="utf-8"), encoding="utf-8")
    (scripts_dir / "judge_up.sh").chmod((scripts_dir / "judge_up.sh").stat().st_mode | stat.S_IEXEC)

    _write_executable(
        venv_bin / "uvicorn",
        """#!/usr/bin/env bash
echo "fake_uvicorn:$LOCALSCRIPT_OLLAMA_MODE:$LOCALSCRIPT_OLLAMA_HOST"
exit 0
""",
    )
    _write_executable(
        venv_bin / "python",
        f"""#!/usr/bin/env bash
if [ "$1" = "-m" ] && [ "$2" = "uvicorn" ]; then
  exec "{(venv_bin / 'uvicorn').resolve()}" "${{@:3}}"
fi
exec "{Path(os.sys.executable).resolve()}" "$@"
""",
    )

    _write_executable(
        fake_bin / "python3",
        f"""#!/usr/bin/env bash
exec "{Path(os.sys.executable).resolve()}" "$@"
""",
    )

    return root, fake_bin


class _TagsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/api/tags":
            self.send_response(404)
            self.end_headers()
            return
        payload = b'{"models":[{"name":"qwen3:8b-q4_K_M"},{"name":"qwen3:4b-instruct-2507-q4_K_M"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


def _start_tags_server():
    server = HTTPServer(("127.0.0.1", 0), _TagsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_judge_up_remote_api_mode_does_not_require_local_ollama(tmp_path):
    root, fake_bin = _prepare_fake_project(tmp_path)
    server = _start_tags_server()

    try:
        completed = subprocess.run(
            ["bash", str(root / "scripts" / "judge_up.sh")],
            cwd=str(root),
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "LOCALSCRIPT_OLLAMA_MODE": "remote_api",
                "LOCALSCRIPT_OLLAMA_HOST": f"http://127.0.0.1:{server.server_port}",
                "LOCALSCRIPT_PYTHON_BIN": str(root / ".venv" / "bin" / "python"),
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    assert completed.returncode == 0
    assert "Ollama mode remote_api" in completed.stdout
    assert f"fake_uvicorn:remote_api:http://127.0.0.1:{server.server_port}" in completed.stdout


def test_judge_up_local_cli_mode_accepts_local_ollama_cli(tmp_path):
    root, fake_bin = _prepare_fake_project(tmp_path)
    server = _start_tags_server()
    _write_executable(
        fake_bin / "ollama",
        """#!/usr/bin/env bash
if [ "$1" = "pull" ]; then
  exit 0
fi
exit 0
""",
    )

    try:
        completed = subprocess.run(
            ["bash", str(root / "scripts" / "judge_up.sh")],
            cwd=str(root),
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "LOCALSCRIPT_OLLAMA_MODE": "local_cli",
                "LOCALSCRIPT_OLLAMA_HOST": f"http://127.0.0.1:{server.server_port}",
                "LOCALSCRIPT_PYTHON_BIN": str(root / ".venv" / "bin" / "python"),
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    assert completed.returncode == 0
    assert "Ollama mode local_cli" in completed.stdout
    assert f"fake_uvicorn:local_cli:http://127.0.0.1:{server.server_port}" in completed.stdout


def test_judge_up_rejects_unsupported_project_python(tmp_path):
    root, fake_bin = _prepare_fake_project(tmp_path)

    completed = subprocess.run(
        ["bash", str(root / "scripts" / "judge_up.sh")],
        cwd=str(root),
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "LOCALSCRIPT_PYTHON_BIN": str(root / ".venv" / "bin" / "python"),
            "LOCALSCRIPT_PYTHON_MIN_MINOR": "99",
            "LOCALSCRIPT_PYTHON_MAX_MINOR": "99",
        },
    )

    assert completed.returncode != 0
    assert "unsupported_python" in completed.stderr
    assert "fake_uvicorn" not in completed.stdout
