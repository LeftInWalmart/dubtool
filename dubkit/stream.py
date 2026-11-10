"""The fast path: analyse a rolling window, dub it, serve it, repeat.

Live mode originally listened to the whole file before it spoke, so the wait before the
first frame scaled with the length - a 40 minute episode spent minutes in transcription,
and no amount of tuning could fix that shape. This module changes the shape instead.

Take WIN seconds of audio, transcribe them, identify who is talking, translate, voice,
mix into the blocks they fall in, and flush. Then move forward HOP seconds and do it
again, always in order, always slightly faster than the viewer consumes it. Only the
first window has to finish before the page can start, so the wait is a constant few
seconds regardless of whether the video is one minute or two hours.

Three things make that possible and each of them is a trade:

* **whisper on a window, not a file.** Beam 1, no word timestamps, language detected
  once on the first window and then pinned. Roughly 0.05x real time for `tiny`, which
  is what a window needs to stay ahead.
* **speakers by embedding, not by clustering the whole file.** Every line is compared
  against the centroids seen so far; a new voice is a new speaker. ECAPA loads in the
  background, so the first window does not wait for it.
* **no separation.** The bed is the original mix with the dialogue pushed down inside
  the spans we have already found (see `ff.duck_bed`). Music and laughter survive,
  a little of the original voice bleeds through, and there is no demucs pass at all.
"""

from __future__ import annotations

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from . import ff
from . import voice as voice_mod
from .live import BLK, BED_GAIN, SPEECH_RMS, SR_MIX
from .pipeline import Config, Work, stage_extract

WIN = 12.0        # seconds of audio looked at per pass
HOP = 8.0         # seconds advanced per pass; WIN-HOP is the overlap that keeps words whole
LOOKAHEAD = 0.15  # a segment must end this far inside the window to be trusted
MIN_LINE = 0.3    # shorter than this is a cough, not a line
SEED = 0.50       # cosine similarity that still counts as the same person


class Asr:
    """One whisper model, reused window after window - loading it per call costs more
    than transcribing the window."""

    def __init__(self, size: str, device: str, lang: str | None = None):
        from faster_whisper import WhisperModel

        compute = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(size, device=device, compute_type=compute)
        self.lang = lang or None
        self.locked = bool(lang)

    def __call__(self, audio: np.ndarray, offset: float) -> list[dict]:
        raw, info = self.model.transcribe(
            audio,
            language=self.lang,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250},
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        if not self.locked:
            self.lang = info.language  # detect once, then stay in that language
            self.locked = True
        out = []
        for s in raw:
            text = (s.text or "").strip()
            if not text:
                continue
            start, end = float(s.start) + offset, float(s.end) + offset
            if end - start < MIN_LINE:
                continue
            out.append({"start": round(start, 3), "end": round(end, 3), "text": text})
        return out


class Speakers:
    """Centroids instead of a global clustering pass: the first window cannot wait for
    diarization, and a viewer only needs every line to land on the same voice as the
    last time that person spoke."""

    def __init__(self) -> None:
        self.cent: list[np.ndarray] = []
        self.ids: list[str] = []
        self.enc = None
        self.lock = threading.Lock()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        try:
            from .speech import _ecapa_encoder

            self.enc = _ecapa_encoder()
        except Exception as exc:  # a missing voice per character is not worth a dead stream
            print(f"      ! speaker identification unavailable ({type(exc).__name__})")

    def assign_many(self, clips: list[np.ndarray]) -> list[str]:
        """One encode_batch for the whole window; padding is fine, ECAPA ignores silence."""
        if self.enc is None or not any(len(c) >= 1600 for c in clips):
            with self.lock:
                return [self.ids[0] if self.ids else self._new() for _ in clips]
        import torch

        width = max(len(c) for c in clips)
        batch = np.zeros((len(clips), width), np.float32)
        for i, c in enumerate(clips):
            batch[i, :len(c)] = c
        with torch.no_grad():
            embs = self.enc.encode_batch(torch.from_numpy(batch)).cpu().numpy()
        out = []
        for e in embs:
            e = e.ravel()
            e = e / (np.linalg.norm(e) + 1e-9)
            with self.lock:
                out.append(self._match(e))
        return out

    def _match(self, e: np.ndarray) -> str:
        if not self.cent:
            return self._new(e)
        sims = [float(e @ c) for c in self.cent]
        j = int(np.argmax(sims))
        if sims[j] >= SEED or len(self.cent) >= 12:
            return self.ids[j]
        return self._new(e)

    def assign(self, clip: np.ndarray) -> str:
        """-> speaker id. Falls back to one voice until ECAPA has loaded."""
        if self.enc is None or len(clip) < 1600:
            with self.lock:
                return self.ids[0] if self.ids else self._new()
        import torch

        with torch.no_grad():
            emb = self.enc.encode_batch(torch.from_numpy(clip).unsqueeze(0))
        e = emb.squeeze().cpu().numpy()
        e = e / (np.linalg.norm(e) + 1e-9)
        with self.lock:
            if not self.cent:
                return self._new(e)
            sims = [float(e @ c) for c in self.cent]
            j = int(np.argmax(sims))
            if sims[j] >= SEED or len(self.cent) >= 12:
                return self.ids[j]
            return self._new(e)

    def _new(self, e: np.ndarray | None = None) -> str:
        sid = f"spk{len(self.ids):02d}"
        self.ids.append(sid)
        if e is not None:
            self.cent.append(e)
        else:  # placeholder so the index stays in step with self.ids
            self.cent.append(np.zeros(192, np.float32))
        return sid


def _duck(gain: np.ndarray, at: float, spans, sr: int, floor: float = 0.06,
          ramp: float = 0.12) -> None:
    """Push the dialogue down inside `spans`, with a ramp so it does not click."""
    n = len(gain)
    mask = np.zeros(n, np.float32)
    for a, b in spans:
        lo = int((a - at) * sr)
        hi = int((b - at) * sr)
        lo, hi = max(0, lo), min(n, hi)
        if hi > lo:
            mask[lo:hi] = 1.0
    r = max(1, int(ramp * sr))
    if mask.any() and r > 1:
        kernel = np.ones(r, np.float32) / r
        mask = np.convolve(mask, kernel, mode="same")
    gain *= 1.0 - (1.0 - floor) * mask


def contain(segs: list[dict], cover: float, end: float) -> tuple[list[dict], float | None]:
    """Which lines of this window are safe to voice, and whether to rewind.

    A line that ends past the window edge was cut in half by the array, so its text is
    incomplete and its timing is a lie: drop it and go back to where it started. Lines
    that begin before `cover` have already been handled by an earlier window.
    """
    keep, rewind = [], None
    for s in segs:
        if s["start"] < cover - 0.2:
            continue
        if s["end"] > end - LOOKAHEAD:
            rewind = s["start"] if rewind is None else min(rewind, s["start"])
            continue
        keep.append(s)
    return keep, rewind


class Streamer:
    def __init__(self, live):
        self.live = live
        self.cfg = live.cfg
        if self.cfg.translate == "auto":
            # a dead OpenRouter key costs 2.3 s of retry ladder before the first window
            # can be voiced, which is the whole point of streaming; Google answers in 0.1 s
            self.cfg.translate = "google"
        self.work: Path = live.work
        self.audio_dir = self.work / "audio"
        self.clips = self.work / "clips"
        self.pending: dict[int, np.ndarray] = {}
        self.written = 0
        self.total = 0.0
        self.n_blocks = 1
        self.spans: list[tuple[float, float]] = []
        self.lines = 0
        self.started = 0.0
        # one pool for the whole run: a fresh pool means fresh threads, and every new
        # thread has to load its own onnx voice (~2 s) before it can speak
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.warm: threading.Thread | None = None

    # -- blocks -------------------------------------------------------------
    def _place(self, clip: np.ndarray, at: float) -> None:
        start = max(0.0, at)
        clip = clip[int((start - at) * SR_MIX):]
        while len(clip) > 0:
            k = int(start // BLK)
            if k >= self.n_blocks:
                return
            if k not in self.pending:
                self.pending[k] = np.zeros((int(SR_MIX * BLK), 2), np.float32)
            pos = int((start - k * BLK) * SR_MIX)
            take = min(len(clip), len(self.pending[k]) - pos)
            if take <= 0:
                return
            self.pending[k][pos:pos + take] += clip[:take]
            clip = clip[take:]
            start += take / SR_MIX

    def _flush(self, upto: int) -> None:
        master = self.work / "master.wav"
        while self.written < min(upto, self.n_blocks):
            k = self.written
            buf = self.pending.pop(k, None)
            if buf is None:
                buf = np.zeros((int(SR_MIX * BLK), 2), np.float32)
            at = k * BLK
            want = min(len(buf), int(max(0.0, self.total - at) * SR_MIX))
            if want > 0:
                with ff.WavReader(str(master)) as r:
                    bed = r.slice(at, at + want / SR_MIX, mono=False)
                if bed.shape[1] == 1:
                    bed = np.repeat(bed, 2, axis=1)
                gain = np.ones(len(bed), np.float32)
                _duck(gain, at, self.spans, SR_MIX)
                buf[:len(bed)] += (bed * gain[:, None] * BED_GAIN)[:len(buf)]
            np.clip(buf, -0.99, 0.99, out=buf)
            fade = int(SR_MIX * 0.008)
            if fade and len(buf) > 2 * fade:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
                buf[:fade] *= ramp
                buf[-fade:] *= ramp[::-1]
            sf.write(str(self.audio_dir / f"{k}.wav"), buf, SR_MIX, subtype="PCM_16")
            self.written += 1
            self.live.set(blocks=self.written, ready=min(self.total, self.written * BLK),
                          rate=(self.written * BLK) / max(0.001, time.time() - self.started))

    # -- the loop -----------------------------------------------------------
    def run(self) -> None:
        live = self.live
        cfg = self.cfg
        live.set(phase="reading the audio")
        master = stage_extract(Work(self.work), _pipe_cfg(live))
        self.total = ff.duration(master)
        live.set(total=self.total)
        self.n_blocks = max(1, math.ceil(self.total / BLK))
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        for old in self.audio_dir.glob("*.wav"):
            old.unlink()
        self.clips.mkdir(exist_ok=True)

        if cfg.tts not in ("piper", "auto"):
            print("      ! live mode is only viable with --tts piper; this engine cannot keep up")

        asr = Asr(cfg.whisper, cfg.device, cfg.source_lang)
        book = Speakers()
        engine = voice_mod.make_engine(cfg.tts, cfg.lang, cfg.device, cfg.tts_model)
        engine.prepare({}, self.work)
        self.warm = threading.Thread(target=self._warm, args=(engine, cfg.lang), daemon=True)
        self.warm.start()
        live.set(phase="dubbing")
        self.started = time.time()

        reader = ff.WavReader(str(master))
        pos, cover, n_win = 0.0, 0.0, 0
        try:
            while pos < self.total - 0.05:
                if getattr(live, "stopped", False):
                    print("      stopped")
                    return
                n_win += 1
                end = min(self.total, pos + WIN)
                t0 = time.time()
                chunk = reader.slice(pos, end, sr=16000)
                if len(chunk) < 1600:
                    break
                t1 = time.time()
                segs = asr(chunk, pos)
                t2 = time.time()
                keep, rewind = contain(segs, cover, end)
                horizon = end - LOOKAHEAD if rewind is None else min(rewind, end - LOOKAHEAD)
                if keep:
                    self._dub(keep, chunk, pos, book, engine, horizon)
                if os.environ.get("DUB_DEBUG"):
                    print(f"    [win {n_win}] read {t1-t0:.2f} asr {t2-t1:.2f} "
                          f"dub {time.time()-t2:.2f} lines {len(keep)}")
                nxt = pos + HOP
                if rewind is not None and rewind > pos:
                    nxt = min(nxt, rewind)
                if nxt <= pos + 0.2:  # never stall on a pathological segment
                    nxt = pos + 0.2
                pos = nxt
                cover = nxt   # everything before here has been voiced or deliberately re-read
            self._flush(self.n_blocks)
        finally:
            reader.close()
            self.pool.shutdown(wait=False)
            try:
                engine.close()
            finally:
                live.set(phase="done", ready=self.total, lines_done=self.lines,
                         lines_total=self.lines)

    def _warm(self, engine, lang: str) -> None:
        """Load the first voices while the audio is still being extracted; onnxruntime
        takes ~2 s per voice and nobody should wait for that in the first block."""
        try:
            voice_for = getattr(engine, "_voice_for", None)
            load = getattr(engine, "_load", None)
            if voice_for and load:
                # every voice this language has (1-3): a load costs 1.5 s and lands in the
                # middle of a window if we wait for the speaker to introduce it
                names = list(getattr(engine, "_available", lambda _l: [])(lang))
                if names:
                    list(self.pool.map(load, names))
        except Exception:
            pass

    def _dub(self, segs: list[dict], chunk: np.ndarray, pos: float,
             book: Speakers, engine, horizon: float) -> None:
        """Voice one window: who, what language it becomes, then the samples."""
        for i, s in enumerate(segs):
            s["id"] = self.lines + i
        t0 = time.time()
        keep = [(s["start"] - pos) * 16000 for s in segs]
        ids = book.assign_many([
            chunk[int(a):int((s["end"] - pos) * 16000)] for s, a in zip(segs, keep)
        ])
        for s, sid in zip(segs, ids):
            s["speaker"] = sid
        t1 = time.time()
        if self.cfg.translate != "none":
            if not self.written:   # once the dub is flowing, a flickering label just looks jittery
                self.live.set(phase=f"translating into {self.cfg.lang}")
            lines, provider = voice_mod.translate_segments(
                segs, self.cfg.source_lang, self.cfg.lang, self.cfg.translate,
                self.cfg.translate_model, str(self.cfg.inp), None, jobs=6,
            )
            for s in segs:
                s["text_tr"] = lines.get(s["id"], s["text"])
        self.live.set(phase="dubbing")
        t2 = time.time()

        def voice(s) -> tuple[dict, Path, Path]:
            slot = max(0.35, s["end"] - s["start"])
            raw = self.clips / f"{s['id']:05d}.wav"
            fitted = self.clips / f"{s['id']:05d}.fit.wav"
            engine.say(s.get("text_tr") or s["text"], {"id": s["speaker"]},
                       self.cfg.lang, raw)
            ff.fit_audio(raw, fitted, slot)
            return s, raw, fitted

        if self.warm is not None:
            # overlap the voice loads with transcription, but never start speaking while a
            # voice is still loading: that is 2-4 s of dead air on the first block
            self.warm.join(120)
            self.warm = None
        workers = max(1, min(getattr(engine, "workers", 1), 4, len(segs)))
        # lazy on purpose: `map` yields in order, so the first line can be placed and its
        # block flushed while the rest of the window is still being voiced
        results = self.pool.map(voice, segs) if workers > 1 else map(voice, segs)
        if os.environ.get("DUB_DEBUG"):
            print(f"      spk {t1-t0:.2f} tr {t2-t1:.2f} lines {len(segs)}")
        starts = [s["start"] for s in segs]
        _tl = time.time()
        for i, (s, _raw, fitted) in enumerate(results):
            if os.environ.get("DUB_DEBUG"):
                print(f"        line {i+1} at {s['start']:.1f}s voiced in {time.time()-_tl:.2f}s")
                _tl = time.time()
            # the next line's start is the first moment that is certainly settled
            nxt = min(horizon, starts[i + 1]) if i + 1 < len(segs) else horizon
            try:
                data, sr = sf.read(str(fitted), dtype="float32", always_2d=True)
            except Exception as exc:  # one bad line must not kill the stream
                print(f"      ! line {s['id']} failed: {exc}")
                self.lines += 1
                continue
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            rms = float(np.sqrt((data ** 2).mean()))
            if rms > 1e-4:
                data = data * (SPEECH_RMS / rms)
            crest = float(np.abs(data).max()) or 1.0
            if crest > 0.85:
                data = data * (0.85 / crest)
            if sr != SR_MIX:
                data = ff.resample(data[:, 0], sr, SR_MIX)[:, None]
                data = np.repeat(data, 2, axis=1)
            self._place(data.astype(np.float32), s["start"])
            self.spans.append((s["start"], s["end"]))
            self.lines += 1
            # everything before the next unvoiced line is finished, so the page may have it
            self._flush(int(max(0.0, nxt) // BLK))
            self.live.set(lines_done=self.lines, lines_total=self.lines)


def _pipe_cfg(live) -> Config:
    """Only the fields `stage_extract` reads; streaming never runs the other stages."""
    c = live.cfg
    return Config(inp=live.src, lang=c.lang, out=live.work / "live.mp4", work=live.work,
                  whisper=c.whisper, tts=c.tts, tts_model=c.tts_model, device=c.device,
                  translate=c.translate, translate_model=c.translate_model,
                  separate="none", jobs=c.jobs, keep_original_audio=True)
