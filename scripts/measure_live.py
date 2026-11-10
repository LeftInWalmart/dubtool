import json, subprocess, sys, time, urllib.request, shutil
from pathlib import Path
args = sys.argv[1:]
src = None
if "--src" in args:
    i = args.index("--src"); src = Path(args[i + 1]); args = args[:i] + args[i + 2:]
name, extra = (args[0], args[1:]) if args else ("run", [])
if src is None:
    src = Path(f"demo/_{name}.mp4"); shutil.copy("demo/excerpt.mp4", src)
# a unique work dir, otherwise a previous run's cached stages make this meaningless
work = Path(f"dubwork/measure-{name}-{int(time.time() * 1000) % 100000}")
proc = subprocess.Popen([r".venv\Scripts\python.exe", "dub.py", "live", str(src), "--lang", "es",
                         "--port", "0", "--buffer", "20", "--no-browser", "--work", str(work), *extra],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
# the server prints the port it actually bound, so a stale one can never answer the poll
base = ""
for line in iter(proc.stdout.readline, ""):
    if "http://localhost:" in line:
        base = line.split("http://localhost:")[1].split("/")[0].strip()
        break
if not base:
    raise SystemExit("server never printed its port")
print(f"  (port {base})", end="  ", flush=True)
t0 = time.time(); seen = {}
try:
    while time.time() - t0 < 420:
        try: st = json.load(urllib.request.urlopen(f"http://127.0.0.1:{base}/state.json", timeout=2))
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
    proc.terminate()
print(" ".join(extra) or "default", "->", json.dumps(seen))
