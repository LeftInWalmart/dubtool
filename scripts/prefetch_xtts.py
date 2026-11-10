"""Chunked, resumable prefetch of the XTTS-v2 weights into the HF cache.

Why: a single 1.87 GB HTTP response stalls on this link around ~480 MB.
100 MB range requests don't. So: fetch the blobs in chunks, verify sha256,
drop them where huggingface_hub expects them, and let TTS find a warm cache.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HUB = Path(os.environ.get("HF_HOME") or Path(__file__).resolve().parent.parent / ".cache" / "huggingface") / "hub"
REPO = "tts-hub/XTTS-v2"
REV = "345b56f6fbe25cca7103f7f34e471b8fe8e4945f"
WANT = ["model.pth", "mel_stats.pth"]
CHUNK = 100 << 20


def api_files():
    url = f"https://huggingface.co/api/models/{REPO}/tree/main"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def download(name: str, oid: str, size: int, dst: Path):
    part = dst.with_suffix(".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:  # bogus leftovers
        part.unlink()
        have = 0
    if have:
        print(f"    {name}: resuming at {have/1e6:.0f} MB")
    url = f"https://huggingface.co/{REPO}/resolve/main/{name}"
    with open(part, "ab") as f:
        while have < size:
            end = min(have + CHUNK, size) - 1
            for attempt in range(1, 6):
                t = time.time()
                p = subprocess.run(
                    ["curl", "-sL", "--max-time", "300", "-r", f"{have}-{end}", url],
                    capture_output=True,
                )
                got = len(p.stdout)
                if p.returncode == 0 and got == end - have + 1:
                    break
                time.sleep(3 * attempt)
            else:
                raise SystemExit(f"{name}: chunk {have}-{end} failed")
            f.write(p.stdout)
            f.flush()
            have += len(p.stdout)
            rate = (end - have + len(p.stdout)) / max(0.001, time.time() - t) / 1e6
            print(f"    {name}: {have/1e6:6.0f}/{size/1e6:.0f} MB  ({rate:.1f} MB/s)", end="\r")
    print()
    h = hashlib.sha256()
    with open(part, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    digest = h.hexdigest()
    if digest != oid:
        raise SystemExit(f"{name}: sha256 mismatch ({digest} != {oid})")
    part.replace(dst)
    print(f"    {name}: ok")


def main():
    model = HUB / f"models--{REPO.replace('/', '--')}"
    blobs, snap = model / "blobs", model / "snapshots" / REV
    snap.mkdir(parents=True, exist_ok=True)
    (model / "refs").mkdir(parents=True, exist_ok=True)
    (model / "refs" / "main").write_text(REV, encoding="utf-8")
    for stale in blobs.glob("*.incomplete"):
        stale.unlink()

    for f in api_files():
        if f["path"] not in WANT:
            continue
        oid = f["lfs"]["oid"]
        blob = blobs / oid
        link = snap / f["path"]
        if blob.exists() and blob.stat().st_size == f["size"]:
            print(f"    {f['path']}: already in cache")
        else:
            download(f["path"], oid, f["size"], blob)
        if not link.exists():
            shutil.copy2(blob, link)  # no symlink rights on Windows
            print(f"    linked {f['path']}")


if __name__ == "__main__":
    sys.exit(main())
