#!/usr/bin/env python
"""dub - analyze a TV episode and re-voice it in another language,
keeping each character's original voice.

  python dub.py doctor
  python dub.py analyze "S01E01.mkv"
  python dub.py run "S01E01.mkv" --lang es
  python dub.py run "S01E01.mkv" --lang th --tts elevenlabs
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dubkit import __version__, configure_cache, ff, pipeline, voice  # noqa: E402

VIDEO_EXT = {".mkv", ".mp4", ".m4v", ".avi", ".mov", ".webm", ".ts", ".mpg", ".mpeg", ".wmv", ".flv"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dub", description="Dub any TV episode into another language, in the original voices.")
    p.add_argument("--version", action="version", version=f"dub {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("inputs", nargs="+", help="video file(s) or a directory of episodes")
        sp.add_argument("--work", help="scratch dir (default: <out>/dubwork/<episode>)")
        sp.add_argument("--whisper", default="large-v3", help="faster-whisper size (default large-v3)")
        sp.add_argument("--source-lang", default=None, help="force source language (default: auto)")
        sp.add_argument("--diarize", default="auto", choices=["auto", "pyannote", "ecapa", "single"],
                        help="speaker splitting (default auto: pyannote if HF_TOKEN else ecapa)")
        sp.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
        sp.add_argument("--cast", help="JSON {speaker_id: character_name} to keep names consistent")
        sp.add_argument("--force", action="store_true", help="ignore cached stage results")

    d = sub.add_parser("doctor", help="check this machine can run dubs")
    d.add_argument("--device", default="auto")

    a = sub.add_parser("analyze", help="language, speakers, transcript - no dubbing")
    common(a)
    a.add_argument("--lang", default="es", help="target language (only used for output paths)")

    r = sub.add_parser("run", help="full dub: analyze -> translate -> re-voice -> mix")
    common(r)
    r.add_argument("--lang", required=True, help="target language, e.g. es fr de ja th")
    r.add_argument("-o", "--out", help="output file (single input only)")
    r.add_argument("--out-dir", default=None, help="output folder (default: next to input)")
    r.add_argument("--tts", default="auto", choices=["auto", "chatterbox", "xtts", "elevenlabs"],
                   help="voice engine (auto: chatterbox when installed, else xtts)")
    r.add_argument("--tts-model", default=None, help="elevenlabs model (default eleven_multilingual_v2)")
    r.add_argument("--jobs", type=int, default=1,
                   help="parallel voice workers (0 = one per two cores; default 1)")
    r.add_argument("--translate", default="auto", choices=["auto", "openrouter", "google", "none"])
    r.add_argument("--translate-model", default="google/gemini-2.5-flash", help="OpenRouter model id")
    r.add_argument("--separate", default="demucs", choices=["demucs", "none"],
                   help="demucs = isolate voices from music (recommended)")
    r.add_argument("--no-subtitles", action="store_true", help="don't write the .srt")
    r.add_argument("--keep-original-audio", action="store_true", help="add original audio as a second track")
    r.add_argument("--delete-voices", action="store_true", help="delete cloned ElevenLabs voices afterwards")
    r.add_argument("--no-loudnorm", action="store_true", help="skip loudness normalisation")

    b = sub.add_parser("batch", help="dub a script: a CSV of lines, no video needed")
    b.add_argument("csv", help="CSV with id, character, text[, audio] columns")
    b.add_argument("--lang", required=True, help="target language, e.g. es fr de ja th")
    b.add_argument("--source-lang", default="auto", help="language of the script (default: auto)")
    b.add_argument("-o", "--out-dir", help="output folder (default: next to the CSV)")
    b.add_argument("--voices", help="folder of <character>.wav reference clips")
    b.add_argument("--voice", help="one reference clip for every character")
    b.add_argument("--tts", default="auto",
                   choices=["auto", "chatterbox", "piper", "xtts", "elevenlabs"])
    b.add_argument("--tts-model", default=None, help="elevenlabs model")
    b.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    b.add_argument("--translate", default="auto", choices=["auto", "openrouter", "google", "none"])
    b.add_argument("--translate-model", default="google/gemini-2.5-flash")
    b.add_argument("--jobs", type=int, default=1,
                   help="parallel voice workers (0 = one per two cores; default 1)")
    b.add_argument("--no-loudnorm", action="store_true")
    b.add_argument("--force", action="store_true")

    lv = sub.add_parser(
        "live",
        help="dub while someone watches: buffers, translates ahead of the playhead, then plays",
    )
    lv.add_argument("source", help="video file, or a URL if yt-dlp is installed")
    lv.add_argument("--lang", required=True, help="target language, e.g. es fr de ja th")
    lv.add_argument("--port", type=int, default=8080)
    lv.add_argument("--buffer", type=float, default=20.0,
                    help="seconds of dub to produce before the viewer may start (default 20)")
    lv.add_argument("--whisper", default="small", help="tiny base small medium large-v3")
    lv.add_argument("--tts", default="piper", choices=["piper", "chatterbox", "xtts", "elevenlabs"],
                    help="piper is the default: it is the only engine faster than playback")
    lv.add_argument("--tts-model", default=None)
    lv.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    lv.add_argument("--translate", default="auto", choices=["auto", "openrouter", "google", "none"])
    lv.add_argument("--translate-model", default="google/gemini-2.5-flash")
    lv.add_argument("--separate", default="none", choices=["demucs", "none"],
                    help="duck the original dialogue (fast, default) or pull it out with demucs")
    lv.add_argument("--jobs", type=int, default=1)
    lv.add_argument("--no-browser", action="store_true", help="don't open a tab")
    lv.add_argument("--work", help="scratch folder (default: dubwork/live-<name>)")
    return p


def expand(inputs) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            files += sorted(x for x in p.iterdir() if x.suffix.lower() in VIDEO_EXT)
        elif any(ch in raw for ch in "*?["):
            files += sorted(p.parent.glob(p.name))
        else:
            files.append(p)
    files = [f for f in files if f.suffix.lower() in VIDEO_EXT]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"not found: {', '.join(map(str, missing))}")
    if not files:
        raise SystemExit("no video files given")
    return files


def cfg_for(args, inp: Path) -> pipeline.Config:
    cast = {}
    if getattr(args, "cast", None):
        cast = json.loads(Path(args.cast).read_text(encoding="utf-8-sig"))
    out = None
    if getattr(args, "out", None):
        out = Path(args.out)
    elif getattr(args, "out_dir", None):
        out = Path(args.out_dir) / f"{inp.stem}.{voice.norm_lang(getattr(args, 'lang', 'es'))}{inp.suffix if inp.suffix.lower() in {'.mp4', '.mkv'} else '.mp4'}"
    return pipeline.Config(
        inp=inp,
        lang=getattr(args, "lang", "es"),
        out=out,
        work=Path(args.work) if args.work else None,
        tts=getattr(args, "tts", "auto"),
        tts_model=getattr(args, "tts_model", None),
        translate=getattr(args, "translate", "auto"),
        translate_model=getattr(args, "translate_model", "google/gemini-2.5-flash"),
        whisper=args.whisper,
        source_lang=args.source_lang,
        diarize=args.diarize,
        device=args.device,
        separate=getattr(args, "separate", "demucs"),
        cast=cast,
        subtitles=not getattr(args, "no_subtitles", False),
        keep_original_audio=getattr(args, "keep_original_audio", False),
        keep_voices=not getattr(args, "delete_voices", False),
        loudnorm=not getattr(args, "no_loudnorm", False),
        jobs=resolve_jobs(getattr(args, "jobs", 1)),
        force=args.force,
    )


def resolve_jobs(n: int) -> int:
    """--jobs 0 means "use the machine": one worker per two cores, capped at 8."""
    if n and n > 0:
        return n
    return max(1, min(8, (os.cpu_count() or 4) // 2))


# -------------------------------------------------------------------- doctor
def doctor(device="auto"):
    ok = True

    def line(good, label, detail=""):
        nonlocal ok
        ok = ok and good
        print(f"  {'OK  ' if good else 'FAIL'}  {label:<26} {detail}")

    print(f"\ndub {__version__}")
    line(sys.version_info >= (3, 10) and sys.version_info < (3, 13),
         "python 3.10-3.12", sys.version.split()[0] + ("" if sys.version_info < (3, 13) else "  <- ML wheels are shaky here"))
    try:
        line(True, "ffmpeg", ff.ffmpeg())
    except SystemExit:
        line(False, "ffmpeg", "winget install --id Gyan.FFmpeg -e")
    for mod, hint in [
        ("torch", "pip install torch torchaudio"),
        ("faster_whisper", "pip install faster-whisper"),
        ("TTS", "pip install coqui-tts"),
        ("demucs", "pip install demucs"),
        ("soundfile", "pip install soundfile"),
        ("sklearn", "pip install scikit-learn"),
        ("speechbrain", "pip install speechbrain  (tokenless diarization)"),
    ]:
        try:
            importlib.import_module(mod)
            line(True, mod)
        except Exception as e:
            line(False, mod, f"{hint}  ({type(e).__name__})")
    try:
        importlib.import_module("pyannote.audio")
        line(True, "pyannote.audio", "installed - best diarization quality")
    except Exception:
        line(True, "pyannote.audio", "not installed - optional, needs HF_TOKEN")
    line(True, "HF_TOKEN", "set" if os.environ.get("HF_TOKEN") else "unset (diarization falls back to ECAPA)")
    line(True, "OPENROUTER_API_KEY", "set" if os.environ.get("OPENROUTER_API_KEY") else "unset (translation falls back to Google)")
    line(True, "ELEVENLABS_API_KEY", "set" if os.environ.get("ELEVENLABS_API_KEY") else "unset (only needed for languages no local engine has)")
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache("tts-hub/XTTS-v2", "model.pth")
        line(isinstance(hit, str), "xtts-v2 weights",
             "cached (non-commercial licence - fine for tests, not for selling)"
             if isinstance(hit, str) else "not cached - only needed for --tts xtts")
    except Exception:
        pass
    try:
        from dubkit.voice import chatterbox_python

        py = chatterbox_python()
        line(True, "chatterbox", f"{py}  (MIT, commercial use ok)")
    except SystemExit:
        line(False, "chatterbox", "not installed - run scripts\\setup_chatterbox.ps1 (MIT engine, needed to sell dubs)")
    try:
        from piper.download_voices import list_voices  # noqa: F401
        import piper

        line(True, "piper", "installed (MIT, ~1.6x real time, needed for live mode)")
    except Exception:
        line(False, "piper", "not installed - .venv\Scripts\python.exe -m pip install piper-tts")
    try:
        import torch

        cuda = torch.cuda.is_available()
        line(True, "gpu", torch.cuda.get_device_name(0) if cuda else "none - CPU works, expect hours per episode")
    except Exception:
        line(False, "gpu", "torch not installed")
    try:
        free = shutil.disk_usage(configure_cache()).free / 1e9
        line(free > 15, "disk (free GB)", f"{free:.0f} GB at {configure_cache()}")
    except Exception:
        pass
    print("\n  " + ("ready to dub.\n" if ok else "install the FAIL lines above, then run doctor again.\n"))
    return 0 if ok else 1


def main(argv=None) -> int:
    # line-buffered + tolerant console: progress shows up live in pipes/logs,
    # and a cp1252 console can't crash us on non-latin dialogue.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, errors="replace")
        except Exception:
            pass
    configure_cache()
    if not (argv if argv is not None else sys.argv[1:]):
        print("""
dub - localize a video or a script, in each character's own voice

  dub.bat doctor                              check this machine
  dub.bat run   "S01E01.mkv" --lang es        dub a finished cut -> S01E01.es.mp4
  dub.bat batch "lines.csv" --voices cast/    dub a script, no video needed
  dub.bat live  "S01E01.mkv" --lang es        watch it while it dubs (opens a browser)

Add --lang with any of: """ + ", ".join(sorted(voice.ALL_LANGS)) + """
Every command takes --help. First time? Run: dub.bat doctor""")
        return 0
    args = build_parser().parse_args(argv)
    if args.cmd == "doctor":
        return doctor(args.device)
    if args.cmd == "live":
        from dubkit import live
        live.serve(live.LiveConfig(
            inp=args.source, lang=args.lang, port=args.port, buffer=args.buffer,
            whisper=args.whisper, tts=args.tts, tts_model=args.tts_model,
            device=args.device, translate=args.translate,
            translate_model=args.translate_model, separate=args.separate,
            jobs=resolve_jobs(args.jobs), open_browser=not args.no_browser,
            work=Path(args.work) if args.work else None))
        return 0
    if args.cmd == "batch":
        from dubkit import batch
        batch.run(batch.Config(
            csv=Path(args.csv), lang=args.lang, source_lang=args.source_lang,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            voices=Path(args.voices) if args.voices else None,
            voice=Path(args.voice) if args.voice else None,
            tts=args.tts, tts_model=args.tts_model, device=args.device,
            translate=args.translate, translate_model=args.translate_model,
            jobs=resolve_jobs(args.jobs), loudnorm=not args.no_loudnorm, force=args.force))
        return 0

    files = expand(args.inputs)
    if args.cmd == "run" and args.out and len(files) > 1:
        raise SystemExit("--out works with one input; use --out-dir for batches")

    rc = 0
    for i, f in enumerate(files, 1):
        if len(files) > 1:
            print(f"\n########## [{i}/{len(files)}] {f.name} ##########")
        cfg = cfg_for(args, f)
        try:
            if args.cmd == "analyze":
                pipeline.analyze(cfg)
            else:
                pipeline.run(cfg)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            rc = 1
            print(f"  FAILED on {f.name}: {type(e).__name__}: {e}")
            if len(files) == 1:
                raise
    return rc


if __name__ == "__main__":
    sys.exit(main())
