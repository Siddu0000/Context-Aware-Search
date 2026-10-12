"""Databricks App entry point: FastAPI on a PRIVATE port, Streamlit on the app port.

Why a launcher: app.yaml's command is not run in a shell, so "uvicorn ... &
streamlit ..." cannot be written there (.../databricks-apps/app-runtime). Databricks'
own app-templates start a backend and a frontend the same way and exit as soon as
EITHER dies -- the platform then shows the app as crashed instead of a Streamlit
page whose every search fails.

Why explicit --host/--port for uvicorn: the runtime sets UVICORN_HOST=0.0.0.0 and
UVICORN_PORT to the APP port (.../databricks-apps/system-env), and the uvicorn CLI
reads UVICORN_* env vars. Without the flags the unauthenticated API would take the
public port. 127.0.0.1 keeps it reachable only from Streamlit in this container.

Local smoke test (needs Databricks auth in the environment):
    DATABRICKS_APP_PORT=8501 python run_app.py
"""

import os
import signal
import subprocess
import sys
import time

BACKEND_PORT = "8001"


def _child_env() -> dict:
    env = dict(os.environ)
    env["BACKEND_URL"] = f"http://127.0.0.1:{BACKEND_PORT}"
    if not env.get("OPENAI_BASE_URL"):
        host = env.get("DATABRICKS_HOST", "").strip().rstrip("/")
        if not host:
            sys.exit("DATABRICKS_HOST is not set: run inside a Databricks App, or set "
                     "OPENAI_BASE_URL to https://<workspace-host>/serving-endpoints.")
        if not host.startswith("http"):
            host = "https://" + host  # the injected value's scheme is not documented
        env["OPENAI_BASE_URL"] = f"{host}/serving-endpoints"
    return env


def main() -> int:
    env = _child_env()
    port = os.environ.get("DATABRICKS_APP_PORT", "8000")
    print(f"run_app: backend 127.0.0.1:{BACKEND_PORT}, UI 0.0.0.0:{port}, "
          f"llm {env.get('OPENAI_MODEL')} at {env.get('OPENAI_BASE_URL')}", flush=True)
    procs = []
    stopping = []

    def stop(*_):
        stopping.append(True)
        for p in procs:
            if p.poll() is None:
                p.terminate()
        deadline = time.time() + 10  # inside the platform's 15 s SIGTERM grace
        for p in procs:
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()

    # Handlers BEFORE the children exist: a stop in between must not orphan them
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # One worker: each loads its own catalog + keyword index (~1.5 GB)
    procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app",
                                   "--host", "127.0.0.1", "--port", BACKEND_PORT,
                                   "--workers", "1", "--no-access-log"], env=env))
    # No access log above and none here: request URLs carry the shopper's query
    procs.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ui.py",
                                   "--server.port", port, "--server.address", "0.0.0.0",
                                   "--server.headless", "true"], env=env))
    while not stopping and all(p.poll() is None for p in procs):
        time.sleep(0.5)
    if stopping:
        print("run_app: stopped by the platform.", flush=True)
        return 0
    died = next(p for p in procs if p.poll() is not None)
    print(f"run_app: {died.args[2]} exited with {died.returncode}; stopping the app.", flush=True)
    stop()
    return died.returncode or 1


if __name__ == "__main__":
    sys.exit(main())
