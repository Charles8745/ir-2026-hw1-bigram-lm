"""簡單計時與 results/timings.json 紀錄。"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


class Stopwatch:
    def __enter__(self):
        self._t0 = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        return False


def record_timing(command: str, seconds: float, path: str | Path = "results/timings.json", **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else []
    data.append({
        "command": command,
        "seconds": round(float(seconds), 3),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    })
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
