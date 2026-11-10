"""Chatterbox TTS worker. Runs in its own interpreter, one process per dub.

Chatterbox pins torch 2.6 / transformers 5.2, which does not coexist with the rest of
DubTool's stack, so the engine lives in a separate venv and talks to us over stdin/stdout.
One process per run keeps the model loaded (a reload is 10-20 s per line otherwise).

Protocol, one JSON object per line, in order:
    -> {"id": 1, "text": "...", "lang": "es", "ref": "C:/.../ref.wav", "out": "C:/.../00001.wav"}
    <- {"id": 1, "ok": true, "seconds": 4.2}
Model chatter goes to stderr; stdout carries replies only. The redirect below is not cosmetic:
chatterbox/diffusers `print()` their warnings, and a single stray line on stdout broke the
protocol (the parent tried to json.loads a warning and died on line 1).
"""

import json
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")  # the sampler bar is noise in a log file

_PROTO = sys.stdout  # keep the real stdout for protocol replies
sys.stdout = sys.stderr  # any library that print()s lands on stderr instead


def send(obj) -> None:
    _PROTO.write(json.dumps(obj) + "\n")
    _PROTO.flush()


def main() -> int:
    import soundfile as sf
    import torch

    threads, device = 0, "cpu"
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            threads = int(argv[i + 1])
        elif a == "--device" and i + 1 < len(argv):
            device = argv[i + 1]
    # Measured on a 12-logical-core laptop: 8 threads beats 4, 6 and 12 (12 is 15% slower than 8).
    # The intra-op pool stops paying past ~2/3 of the cores; the rest only adds contention.
    torch.set_num_threads(threads if threads > 0 else max(1, (os.cpu_count() or 4) * 2 // 3))

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    sr = model.sr
    send({"ready": True, "sr": sr, "device": device})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        job = json.loads(line)
        if job.get("stop"):
            break
        t0 = time.time()
        try:
            kw = {"language_id": job["lang"]}
            if job.get("ref"):
                kw["audio_prompt_path"] = job["ref"]
            for key in ("exaggeration", "cfg_weight", "temperature"):
                if job.get(key) is not None:
                    kw[key] = job[key]
            with torch.inference_mode():
                wav = model.generate(job["text"], **kw)
            sf.write(job["out"], wav.squeeze(0).cpu().numpy(), sr, subtype="PCM_16")
            reply = {"id": job.get("id"), "ok": True, "seconds": round(time.time() - t0, 2)}
        except Exception as exc:  # noqa: BLE001 - report and keep the worker alive
            reply = {"id": job.get("id"), "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        send(reply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
