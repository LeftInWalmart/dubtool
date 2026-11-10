import json, subprocess, sys, time, urllib.request, shutil
from pathlib import Path
name, extra = sys.argv[1], sys.argv[2:]
src = Path(f"demo/_{name}.mp4"); shutil.copy("demo/excerpt.mp4", src)
work = Path(f"dubwork/live-_{name}"); shutil.rmtree(work, ignore_errors=True)
p = subprocess.Popen([r".venv\Scripts\python.exe", "dub.py", "live", str(src), "--lang", "es",
                      "--port", "8105", "--buffer", "20", "--no-browser", *extra],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
t0 = time.time(); seen = {}
try:
    while time.time() - t0 < 420:
        try: st = json.load(urllib.request.urlopen("http://127.0.0.1:8105/state.json", timeout=2))
        except Exception: time.sleep(0.4); continue
        if st["phase"] not in seen:
            seen[st["phase"]] = round(time.time() - t0, 1)
        need = min(st["buffer"], max(4, st["total"] * .5))
        if "PLAY" not in seen and st["ready"] >= need:
            seen["PLAY"] = round(time.time() - t0, 1)
        if st["phase"] == "done":
            seen["done"] = round(time.time() - t0, 1); break
        time.sleep(0.5)
finally:
    p.terminate()
print(" ".join(extra) or "default", "->", json.dumps(seen))
