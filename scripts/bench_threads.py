"""Thread-count sweep for the voice worker on this machine.

Parallel *workers* do not help on this laptop (memory-bandwidth bound), so the only
CPU lever left is intra-op threads. Runs the same lines with different `--threads`
values, each in its own worker process, and reports seconds per line.

    .venv\\Scripts\\python.exe scripts\\bench_threads.py --lines 3 --threads 12 8 6 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dubkit import voice  # noqa: E402

WORK = Path("demo/dubwork/excerpt")
OUT = Path("demo/_bench_threads")


def load_lines(n: int):
    data = json.loads((WORK / "translate.es.json").read_text(encoding="utf-8"))
    speakers = json.loads((WORK / "speakers.json").read_text(encoding="utf-8"))["speakers"]
    segs = [s for s in data["segments"] if (s.get("text_tr") or "").strip()][:n]
    return [s["text_tr"].strip() for s in segs], speakers


def run(threads: int, texts: list[str], speakers: dict) -> tuple[float, float, int]:
    OUT.mkdir(parents=True, exist_ok=True)
    py = voice.chatterbox_python()
    cmd = [py, str(Path(voice.__file__).parent / "engines" / "chatterbox_worker.py"),
           "--device", "cpu", "--threads", str(threads)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
    for line in proc.stdout:
        if line.strip().startswith("{"):
            json.loads(line)
            break
    spk = next(iter(speakers.values()))
    t0 = time.time()
    speech = 0.0
    for i, text in enumerate(texts):
        out = OUT / f"t{threads}_{i}.wav"
        proc.stdin.write(json.dumps({"id": i, "text": text, "lang": "es",
                                     "ref": spk.get("ref"), "out": str(out)}) + "\n")
        proc.stdin.flush()
        for line in proc.stdout:
            if not line.strip().startswith("{"):
                continue
            reply = json.loads(line)
            if reply.get("id") != i:
                continue
            if reply.get("ok"):
                speech += voice.sf.info(str(out)).duration
            break
    wall = time.time() - t0
    proc.stdin.write('{"stop":1}\n')
    proc.stdin.flush()
    proc.wait(timeout=60)
    return wall, speech, len(texts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=3)
    ap.add_argument("--threads", type=int, nargs="+", default=[12, 8, 6, 4])
    args = ap.parse_args()

    texts, speakers = load_lines(args.lines)
    print(f"{len(texts)} lines, {len(texts)} generated per config\n")
    rows = []
    for threads in args.threads:
        wall, speech, n = run(threads, texts, speakers)
        rows.append((threads, wall))
        print(f"  threads={threads:2d}: total {wall:6.1f}s  "
              f"{wall / n:5.1f}s per line  RTF {wall / max(0.1, speech):5.2f}x")
    best = min(rows, key=lambda r: r[1])
    print(f"\n  best: threads={best[0]} ({best[1]:.1f}s)  "
          f"{rows[0][1] / best[1]:.2f}x faster than threads={rows[0][0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
