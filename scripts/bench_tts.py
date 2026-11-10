"""How fast can the voice stage go on this machine?

Times the same N lines with different `--jobs` values. Worker start-up (model load,
~20 s each) is included, printed separately, because it dominates short runs.

    .venv\\Scripts\\python.exe scripts\\bench_tts.py --lines 12 --jobs 1 2 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dubkit import voice  # noqa: E402

WORK = Path("demo/dubwork/excerpt")
OUT = Path("demo/_bench")


def load_lines(n: int):
    data = json.loads((WORK / "translate.es.json").read_text(encoding="utf-8"))
    # speakers.json is {method, speakers, segments, turns} — the "/"speakers"/" map is what
    # the engine wants, and prepare() needs it to pick reference clips.
    speakers = json.loads((WORK / "speakers.json").read_text(encoding="utf-8"))["speakers"]
    segs = [s for s in data["segments"] if (s.get("text_tr") or "").strip()][:n]
    items = []
    for i, s in enumerate(segs):
        spk = speakers.get(s["speaker"]) or next(iter(speakers.values()))
        items.append((s["text_tr"].strip(), spk, "es", OUT / f"j{i:03d}.wav"))
    return items, speakers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=12)
    ap.add_argument("--jobs", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    items, speakers = load_lines(args.lines)
    audio = sum(1 for _ in items)
    print(f"{len(items)} lines, device={args.device}\n")

    rows = []
    for jobs in args.jobs:
        shutil.rmtree(OUT, ignore_errors=True)
        OUT.mkdir(parents=True, exist_ok=True)
        eng = voice.ChatterboxEngine(device=args.device, jobs=jobs)
        t_load = time.time()
        eng.prepare(speakers, WORK)  # starts one worker; say_many adds the rest
        load = time.time() - t_load
        t0 = time.time()
        eng.say_many(items)
        wall = time.time() - t0
        eng.close()
        ok = sum(1 for *_, out in items if Path(out).exists() and Path(out).stat().st_size > 512)
        speech = sum(voice.sf.info(str(out)).duration for *_, out in items if Path(out).exists())
        rows.append((jobs, load, wall, ok, speech))
        print(f"  jobs={jobs}: first-load {load:5.1f}s  total {wall:6.1f}s  "
              f"lines {ok}/{audio}  ({wall / max(1, ok):5.1f}s per line, "
              f"RTF {wall / max(0.1, speech):5.1f}x)")
        for err in eng.failures[:3]:
            print(f"    failed {err}")

    if len(rows) > 1:
        base = rows[0][2]
        print("\n  speed-up vs jobs=1: " + "  ".join(f"jobs={r[0]}: {base / r[2]:.2f}x" for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
