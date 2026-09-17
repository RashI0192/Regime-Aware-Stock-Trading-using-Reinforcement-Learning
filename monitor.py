"""Basic deployment health monitor. Usage: python monitor.py http://localhost:8000"""
import sys
from urllib.request import urlopen

base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"
try:
    with urlopen(base + "/health", timeout=10) as response:
        body = response.read().decode()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        print(f"OK {base}/health {body}")
except Exception as exc:
    print(f"ALERT {base}/health: {exc}")
    raise SystemExit(1)
