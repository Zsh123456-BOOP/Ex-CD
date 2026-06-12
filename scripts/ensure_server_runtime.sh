#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$HOME/anaconda3/envs/xph_env/bin/python}"

"$PYTHON_BIN" - <<'PY'
import importlib.util
import subprocess
import sys

missing = [name for name in ("pyarrow",) if importlib.util.find_spec(name) is None]
if missing:
    print("[INFO] Installing required export dependency:", ",".join(missing), flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
else:
    print("[OK] runtime export dependencies available")
PY
