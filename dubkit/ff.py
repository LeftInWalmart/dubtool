"""ffmpeg + wav helpers. Everything media-shaped lives here."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

_CACHE: dict[str, str] = {}
_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def ffmpeg() -> str:
    """Prefer PATH ffmpeg, else the binary bundled by imageio-ffmpeg."""
    if "exe" not in _CACHE:
        exe = shutil.which("ffmpeg") or ""
        if not exe:
            try:
                import imageio_ffmpeg

                exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                exe = ""
        if not exe or not Path(exe).exists():
            raise SystemExit(
                "ffmpeg not found.\n"
                "  winget install --id Gyan.FFmpeg -e\n"
                "  or: pip install imageio-ffmpeg"
            )
        _CACHE["exe"] = exe
    return _CACHE["exe"]


def run(cmd) -> str:
    p = subprocess.run(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if p.returncode != 0:
        tail = "\n".join((p.stdout or "").strip().splitlines()[-15:])
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}\n{tail}")
    return p.stdout or ""


def _ffmpeg_stderr(path) -> str:
    p = subprocess.run(
        [ffmpeg(), "-hide_banner", "-i", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return p.stdout or ""


def duration(path) -> float:
    path = Path(path)
    if path.suffix.lower() == ".wav":
        return float(sf.info(str(path)).duration)
    probe = shutil.which("ffprobe")
    if probe:
        p = subprocess.run(
            [probe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            return float(json.loads(p.stdout)["format"]["duration"])
        except Exception:
            pass
    m = _DUR_RE.search(_ffmpeg_stderr(path))
    if not m:
        raise RuntimeError(f"cannot read duration of {path}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def has_video(path) -> bool:
    return "Video:" in _ffmpeg_stderr(path)


def extract_audio(src, dst, sr: int = 16000, channels: int = 1) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
            "-vn", "-ac", str(channels), "-ar", str(sr), "-c:a", "pcm_s16le", str(dst),
        ]
    )
    return dst


def fit_audio(src, dst, target_sec: float, ar: int = 44100, max_rate: float = 1.5,
              trim: bool = True) -> Path:
    """Trim dead air, then time-stretch src to fit target_sec, then resample to ar mono.

    Two passes on purpose: the stretch factor has to be measured *after* the padding is
    gone, otherwise every clip is over-compressed by however much silence the engine added
    and the line ends early. `atempo` is capped - extreme stretching sounds like a robot,
    so a bad fit stays a bad fit (documented ceiling, not a bug).
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cut = "silenceremove=start_periods=1:start_threshold=-40dB:start_silence=0.08"
    source = Path(src)
    if trim:
        # -40 dB, keep 80 ms: drops engine padding, spares the attack of the first phoneme
        trimmed = dst.with_suffix(".trim.wav")
        run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-vn",
             "-filter:a", f"{cut},areverse,{cut},areverse", "-ac", "1", "-c:a", "pcm_s16le", str(trimmed)])
        if duration(trimmed) >= 0.1:
            source = trimmed
    have = duration(source)
    factor = have / target_sec if target_sec > 0.05 and have > 0.05 else 1.0
    factor = min(max(factor, 1.0 / max_rate), max_rate)
    chain = [] if abs(factor - 1.0) <= 0.03 else [f"atempo={factor:.4f}"]
    chain.append(f"aresample={ar}")
    run(
        [
            ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-vn", "-filter:a", ",".join(chain), "-ac", "1", "-c:a", "pcm_s16le", str(dst),
        ]
    )
    if source != Path(src):
        source.unlink(missing_ok=True)
    return dst


def duck_mix(background, vocals, out, spans, ramp: float = 0.12, block: int = 44100) -> Path:
    """background + vocals, with the vocals muted inside `spans` (the dialogue).

    Why not just use the separator's `no_vocals` stem: on real broadcast audio the
    separator routinely dumps everything into `vocals` (measured on a 1961 print:
    no_vocals came out 31 dB below the mix, i.e. silent). Muting the whole stem then
    deletes the laugh track along with the dialogue and the dub plays over nothing.

    This keeps `background` everywhere and keeps whatever is in `vocals` between lines
    - laughter, applause, room tone - while removing only the speech itself.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    spans = sorted((max(0.0, float(a)), max(0.0, float(b))) for a, b in spans if b > 0)
    with sf.SoundFile(str(background)) as bg, sf.SoundFile(str(vocals)) as vc, sf.SoundFile(
        str(out), "w", samplerate=bg.samplerate, channels=bg.channels, subtype="PCM_16"
    ) as o:
        sr = bg.samplerate
        if (vc.samplerate, vc.channels) != (sr, bg.channels):
            raise SystemExit("duck_mix: stems differ in rate/channels - fix the extraction first")
        pos = 0
        while True:
            bed = bg.read(block, dtype="float32", always_2d=True)
            if not len(bed):
                break
            voc = vc.read(len(bed), dtype="float32", always_2d=True)
            if len(voc) < len(bed):
                voc = np.pad(voc, ((0, len(bed) - len(voc)), (0, 0)))
            t0, t1 = pos / sr, (pos + len(bed)) / sr
            gain = np.ones(len(bed), dtype="float32")
            for start, end in spans:
                if end + ramp < t0 or start - ramp > t1:
                    continue
                t = (pos + np.arange(len(bed))) / sr
                g = 1.0 - np.minimum((t - (start - ramp)) / ramp, (end + ramp - t) / ramp)
                gain *= np.clip(g, 0.0, 1.0).astype("float32")
            o.write(bed + voc * gain[:, None])
            pos += len(bed)
    return out


def duck_bed(src, out, spans, floor: float = 0.06, ramp: float = 0.12,
             block: int = 44100) -> Path:
    """The mix itself, with the dialogue pushed down instead of separated out.

    Live mode uses this when the viewer would rather start in twenty seconds than wait a
    minute for demucs: music, laughter and effects are untouched, the original voice is
    dropped ~24 dB under the dub. Some bleed gets through - that is the trade.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    spans = sorted((max(0.0, float(a)), max(0.0, float(b))) for a, b in spans if b > 0)
    with sf.SoundFile(str(src)) as f, sf.SoundFile(
        str(out), "w", samplerate=f.samplerate, channels=f.channels, subtype="PCM_16"
    ) as o:
        sr = f.samplerate
        pos = 0
        while True:
            buf = f.read(block, dtype="float32", always_2d=True)
            if not len(buf):
                break
            t0, t1 = pos / sr, (pos + len(buf)) / sr
            gain = np.ones(len(buf), dtype="float32")
            for start, end in spans:
                if end + ramp < t0 or start - ramp > t1:
                    continue
                t = (pos + np.arange(len(buf))) / sr
                dip = np.minimum((t - (start - ramp)) / ramp, (end + ramp - t) / ramp)
                gain = np.minimum(gain, 1.0 - (1.0 - floor) * np.clip(dip, 0.0, 1.0))
            o.write(buf * gain[:, None])
            pos += len(buf)
    return out


def mix(background, dub, out, loudnorm: bool = True) -> Path:
    """Background bed + dubbed speech, single pass."""
    out = Path(out)
    chain = "[0:a][1:a]amix=inputs=2:duration=first:normalize=0"
    if loudnorm:
        chain += ",loudnorm=I=-16:TP=-1.5:LRA=11"
    run(
        [
            ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(background), "-i", str(dub),
            "-filter_complex", chain,
            "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(out),
        ]
    )
    return out


def mux(video, audio, out, extra_audio=(), subtitle=None) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(video), "-i", str(audio)]
    for a in extra_audio:
        cmd += ["-i", str(a)]
    for s in ([subtitle] if subtitle else []):
        cmd += ["-i", str(s)]
    cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    for i in range(len(extra_audio)):
        cmd += ["-map", f"{i + 2}:a:0"]
    if subtitle:
        cmd += ["-map", f"{2 + len(extra_audio)}:s:0"]
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    ext = out.suffix.lower()
    if subtitle:
        # mp4/mov only carry mov_text; everything else takes srt
        cmd += ["-c:s", "mov_text" if ext in (".mp4", ".m4v", ".mov") else "srt"]
    if ext in (".mp4", ".m4v", ".mov"):
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(out))
    run(cmd)
    return out


class WavReader:
    """Random-access mono reader: seek + read, no giant arrays."""

    def __init__(self, path):
        self.path = str(path)
        self.f = sf.SoundFile(self.path)
        self.sr = self.f.samplerate
        self.channels = self.f.channels

    def slice(self, start: float, end: float, sr: int | None = None, mono: bool = True) -> np.ndarray:
        start = max(0.0, float(start))
        n = max(0, int(round((float(end) - start) * self.sr)))
        self.f.seek(min(int(round(start * self.sr)), len(self.f)))
        data = self.f.read(n, dtype="float32", always_2d=True)
        if mono and data.shape[1] > 1:
            data = data.mean(axis=1, keepdims=True)
        if sr and sr != self.sr and len(data):
            data = resample(data[:, 0], self.sr, sr)[:, None]
        return data[:, 0] if mono else data

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Cheap linear resample. Only used for embeddings, never for the mix."""
    if sr_in == sr_out or len(x) == 0:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    if n_out < 1:
        return np.zeros(1, dtype=np.float32)
    xp = np.linspace(0.0, 1.0, len(x), endpoint=False)
    xq = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(xq, xp, x).astype(np.float32)


def write_wav(path, data: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(data, dtype=np.float32), sr, subtype="PCM_16")
    return path
