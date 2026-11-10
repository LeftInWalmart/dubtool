"""Watch a video in another language while it is still being dubbed.

The trick that makes this possible is `--tts piper`: it runs at ~1.6x real time on one CPU
core, so the dub is produced faster than the viewer consumes it. Everything else here is
plumbing to keep the picture behind the audio and to stall honestly instead of drifting.

    dub.py live "S01E01.mkv" --lang es

Phases, all reported to the page so it can say what it is doing:

    1. remux   strip the audio, keep the picture (browser-playable mp4)
    2. prepare extract / separate / transcribe / identify speakers / translate   [once, cached]
    3. render  synthesise lines in time order and cut the mixed audio into 5 s blocks
    4. play    the page schedules each block on the WebAudio clock and slaves the video to it
"""

from __future__ import annotations

import json
import math
import mimetypes
import re
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import soundfile as sf

from . import ff
from .pipeline import SR_MIX, Config, Work, stage_asr, stage_bed, stage_extract, stage_speakers, stage_split

BLK = 5.0          # seconds of audio per block: the player's unit of buffering
FADE = 0.008       # 8 ms at each block edge, or the joins click
SPEECH_RMS = 0.10  # -20 dBFS: piper and chatterbox disagree wildly about level
BED_GAIN = 0.70    # music and effects sit under the dialogue, not next to it


@dataclass
class LiveConfig:
    inp: str
    lang: str
    port: int = 8080
    buffer: float = 20.0
    whisper: str = "small"
    tts: str = "piper"
    tts_model: str | None = None
    device: str = "cpu"
    translate: str = "auto"
    translate_model: str | None = None
    separate: str = "demucs"
    jobs: int = 1
    open_browser: bool = True
    work: Path | None = None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "stream"


def resolve_source(inp: str, work: Path) -> Path:
    """A URL becomes a local file via yt-dlp, because the whole pipeline needs random access."""
    if not re.match(r"^https?://", inp):
        path = Path(inp)
        if not path.exists():
            raise SystemExit(f"no such file: {path}")
        return path
    work.mkdir(parents=True, exist_ok=True)
    existing = sorted(work.glob("source.*"))
    if existing:
        return existing[0]
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise SystemExit(
            "a URL needs yt-dlp:  pip install yt-dlp\n"
            "(or download the file yourself and pass the path)"
        ) from None
    print(f"  downloading {inp}")
    out = work / "source.%(ext)s"
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format", "mp4", "-o", str(out), inp,
    ]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"yt-dlp failed (exit {rc})")
    found = sorted(work.glob("source.*"))
    if not found:
        raise SystemExit("yt-dlp produced nothing")
    return found[0]


class Live:
    """Owns the state the browser polls, and the thread that renders blocks."""

    def __init__(self, cfg: LiveConfig):
        self.cfg = cfg
        self.phase = "starting"
        self.error: str | None = None
        self.title = Path(cfg.inp).stem or cfg.inp
        self.total = 0.0
        self.ready = 0.0
        self.blocks = 0
        self.lines_done = 0
        self.lines_total = 0
        self.rate = 0.0            # seconds of dub produced per second of wall clock
        self.out_dir = Path(".")
        self.lock = threading.Lock()

    # -- state the page reads ------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            return {
                "phase": self.phase,
                "error": self.error,
                "title": self.title,
                "lang": self.cfg.lang,
                "total": round(self.total, 2),
                "ready": round(self.ready, 2),
                "blocks": self.blocks,
                "lines_done": self.lines_done,
                "lines_total": self.lines_total,
                "rate": round(self.rate, 2),
                "buffer": self.cfg.buffer,
                "engine": self.cfg.tts,
                "block_seconds": BLK,
            }

    def set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    # -- phase 1: picture ----------------------------------------------------
    def remux(self, src: Path, work: Path) -> Path:
        """mp4 the browser can actually play, with the audio thrown away."""
        out = work / "picture.mp4"
        if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
            return out
        self.set(phase="remux")
        base = [ff.ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
                "-an", "-sn", "-dn", "-movflags", "+faststart", "-pix_fmt", "yuv420p"]
        for codec in (["-c:v", "copy"], ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]):
            rc = subprocess.run(base + codec + [str(out)]).returncode
            if rc == 0 and out.exists() and out.stat().st_size > 1024:
                return out
        raise SystemExit("could not make a web-playable video from that file")

    # -- phase 2: the same stages a normal run uses, cached -----------------
    def prepare(self, src: Path, root: Path) -> tuple[dict, list, Path, float]:
        work = Work(root)
        cfg = Config(inp=src, lang=self.cfg.lang, out=root / "live.mp4", work=root,
                     whisper=self.cfg.whisper, tts=self.cfg.tts, tts_model=self.cfg.tts_model,
                     device=self.cfg.device, translate=self.cfg.translate,
                     translate_model=self.cfg.translate_model, separate=self.cfg.separate,
                     jobs=self.cfg.jobs, keep_original_audio=True)

        self.set(phase="audio")
        total = ff.duration(src)
        self.set(total=total)
        master = stage_extract(work, cfg)
        vocals, novocals = stage_split(work, cfg, total)

        self.set(phase="transcribing")
        asr = stage_asr(work, cfg, vocals)
        segments = asr["segments"]

        self.set(phase="identifying voices")
        speakers = stage_speakers(work, cfg, vocals, segments)["speakers"]

        tr = work.p(f"translate.{self.cfg.lang}.json")
        if tr.exists():
            segments = json.loads(tr.read_text(encoding="utf-8"))["segments"]
        else:
            self.set(phase=f"translating into {self.cfg.lang}")
            from . import voice as voice_mod

            lines, provider = voice_mod.translate_segments(
                segments, asr["lang"], self.cfg.lang, self.cfg.translate,
                self.cfg.translate_model, src.stem, None,
            )
            for seg in segments:
                seg["text_tr"] = lines.get(seg["id"], seg["text"])
            tr.write_text(json.dumps({"lang": self.cfg.lang, "provider": provider,
                                      "segments": segments}, ensure_ascii=False), encoding="utf-8")
        bed = stage_bed(work, vocals, novocals, segments)
        self.set(lines_total=len(segments))
        return speakers, segments, bed, total

    # -- phase 3: render blocks --------------------------------------------
    def render(self, speakers: dict, segments: list, bed: Path, total: float) -> None:
        from . import voice as voice_mod

        audio_dir = self.out_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        for old in audio_dir.glob("*.wav"):
            old.unlink()

        engine = voice_mod.make_engine(self.cfg.tts, self.cfg.lang, self.cfg.device,
                                       self.cfg.tts_model, jobs=self.cfg.jobs)
        engine.prepare(speakers, self.out_dir)
        clips = self.out_dir / "clips"
        clips.mkdir(exist_ok=True)

        n_blocks = max(1, math.ceil(total / BLK))
        pending = {k: np.zeros((int(SR_MIX * BLK), 2), np.float32) for k in range(min(n_blocks, 4))}
        written = 0
        started = time.time()

        def flush(upto_block: int):
            """Write every block below `upto_block` and stop holding its memory."""
            nonlocal written
            while written < min(upto_block, n_blocks):
                if written not in pending:
                    pending[written] = np.zeros((int(SR_MIX * BLK), 2), np.float32)
                buf = pending.pop(written)
                self.add_bed(buf, bed, written, total)
                np.clip(buf, -0.99, 0.99, out=buf)  # backstop: a block must never clip
                fade = int(SR_MIX * FADE)
                if fade and len(buf) > 2 * fade:
                    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
                    buf[:fade] *= ramp
                    buf[-fade:] *= ramp[::-1]
                sf.write(str(audio_dir / f"{written}.wav"), buf, SR_MIX, subtype="PCM_16")
                written += 1
                # measured in seconds of finished audio per second of wall clock, so a viewer
                # can be told whether this machine is actually keeping up
                self.set(blocks=written, ready=min(total, written * BLK),
                         rate=(written * BLK) / max(0.001, time.time() - started))

        def place(clip: np.ndarray, at: float):
            """Add a clip into the blocks it spans, dropping the part past the end."""
            start = max(0.0, at)
            offset = (start - at) * SR_MIX
            clip = clip[int(offset):]
            while len(clip) > 0:
                k = int(start // BLK)
                if k >= n_blocks:
                    return
                if k not in pending:
                    pending[k] = np.zeros((int(SR_MIX * BLK), 2), np.float32)
                pos = int((start - k * BLK) * SR_MIX)
                take = min(len(clip), len(pending[k]) - pos)
                if take <= 0:
                    return
                pending[k][pos:pos + take] += clip[:take]
                clip = clip[take:]
                start += take / SR_MIX

        ordered = sorted(segments, key=lambda s: s["start"])
        for i, seg in enumerate(ordered):
            slot = max(0.35, seg["end"] - seg["start"])
            raw = clips / f"{seg['id']:04d}.wav"
            fitted = clips / f"{seg['id']:04d}.fit.wav"
            if not fitted.exists():
                try:
                    engine.say(seg.get("text_tr") or seg["text"], speakers.get(seg["speaker"], {}),
                               self.cfg.lang, raw)
                    ff.fit_audio(raw, fitted, slot)
                except Exception as exc:  # a missing line is better than a dead stream
                    print(f"      ! line {seg['id']} failed: {exc}")
                    self.set(lines_done=i + 1)
                    continue
            data, sr = sf.read(str(fitted), dtype="float32", always_2d=True)
            if data.shape[1] == 1:  # fit_audio hands back mono at the mix rate
                data = np.repeat(data, 2, axis=1)
            # Engines hand back a full-scale peak with a 0.18-0.31 RMS, which clips the moment
            # the bed is added. Even out by RMS, then cap the crest.
            rms = float(np.sqrt((data ** 2).mean()))
            if rms > 1e-4:
                data = data * (SPEECH_RMS / rms)
            crest = float(np.abs(data).max()) or 1.0
            if crest > 0.85:
                data = data * (0.85 / crest)
            if sr != SR_MIX:
                data = ff.resample(data, sr, SR_MIX)
            place(data, seg["start"])
            self.set(lines_done=i + 1)

            nxt = ordered[i + 1]["start"] if i + 1 < len(ordered) else total
            flush(int(nxt // BLK))
            # how far ahead of the viewer are we?
            self.set(total=total)

        flush(n_blocks)
        engine.close()
        self.set(phase="done", ready=total)

    def add_bed(self, buf: np.ndarray, bed: Path, k: int, total: float) -> None:
        at = k * BLK
        want = min(len(buf), int(max(0.0, total - at) * SR_MIX))
        if want <= 0:
            return
        try:
            with ff.WavReader(str(bed)) as r:
                seg = r.slice(at, at + want / SR_MIX)
                if seg.ndim == 1:
                    seg = np.stack([seg, seg], axis=1)
                if seg.shape[1] == 1:
                    seg = np.repeat(seg, 2, axis=1)
                buf[:len(seg)] += (seg[:len(buf)] * BED_GAIN).astype(np.float32)
        except Exception as exc:
            print(f"      ! bed block {k} unreadable ({exc}); continuing without ambience")

    # -- the whole thing ----------------------------------------------------
    def run(self) -> None:
        src = resolve_source(self.cfg.inp, Path("dubwork") / f"live-{_slug(self.cfg.inp)}")
        work = self.cfg.work or Path("dubwork") / f"live-{src.stem}"
        work.mkdir(parents=True, exist_ok=True)
        self.out_dir = work
        picture = self.remux(src, work)
        self.picture = picture
        speakers, segments, bed, total = self.prepare(src, work)
        self.set(phase="dubbing")
        self.render(speakers, segments, bed, total)


class Handler(BaseHTTPRequestHandler):
    live: Live
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the console for progress, not access logs
        pass

    def handle_one_request(self):
        # a viewer closing the tab mid-block is normal, not a traceback on the user's console
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    # -- helpers ------------------------------------------------------------
    def _json(self, obj: dict, code: int = 200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str | None = None):
        """Single-range file serving; browsers will not seek a video without it."""
        if not path.exists():
            return self._json({"error": "not ready"}, 404)
        size = path.stat().st_size
        ctype = ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        partial = False
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                partial = True
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
                if not m.group(1) and m.group(2):  # suffix range
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(262144, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    # -- routes -------------------------------------------------------------
    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/":
            page = Path(__file__).with_name("player.html").read_text(encoding="utf-8")
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/state.json":
            return self._json(self.live.snapshot())
        if path in ("/video", "/video.mp4"):
            return self._file(getattr(self.live, "picture", Path("nope.mp4")), "video/mp4")
        m = re.fullmatch(r"/audio/(\d+)\.wav", path)
        if m:
            return self._file(self.live.out_dir / "audio" / f"{int(m.group(1))}.wav", "audio/wav")
        return self._json({"error": "no route"}, 404)


def serve(cfg: LiveConfig) -> None:
    live = Live(cfg)
    Handler.live = live

    def work():
        try:
            live.run()
        except SystemExit as exc:
            live.set(phase="error", error=str(exc))
            print(f"\n!! {exc}")
        except Exception as exc:  # the page must hear about it, not just the terminal
            live.set(phase="error", error=f"{type(exc).__name__}: {exc}")
            print(f"\n!! {type(exc).__name__}: {exc}")

    url = f"http://localhost:{cfg.port}/"
    server = ThreadingHTTPServer(("127.0.0.1", cfg.port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=work, daemon=True).start()
    print(f"  watching {cfg.inp}  ->  {cfg.lang}  ({cfg.tts})")
    print(f"  open {url}  (ctrl-c to stop)")
    if cfg.open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  stopped")
        server.shutdown()
