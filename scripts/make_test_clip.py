"""Build a tiny 2-speaker 'episode' for smoke-testing the pipeline.

Keeps the same shape as a real file: dialogue over background music, video stream.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent.parent / "_smoke"
SR = 44100


def ff() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def say(text: str, voice: str, out: Path):
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SelectVoice('{voice}'); $s.Rate = 0; "
        f"$s.SetOutputToWaveFile('{out}'); "
        f"$s.Speak('{text}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def to44(src: Path, dst: Path):
    subprocess.run(
        [ff(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
         "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    david, zira = "Microsoft David Desktop", "Microsoft Zira Desktop"
    lines = [
        (david, "Listen to me. We only have one shot at this, and I am not going to waste it."),
        (zira, "You always say that. Last time we ended up stuck in a basement for three days."),
        (david, "That was different. The basement had a door, and I had a plan."),
        (zira, "The door was painted on the wall, David."),
        (david, "Fine. This time I brought a key."),
    ]
    gap = np.zeros(int(0.7 * SR), np.float32)
    voice, track = [], []
    for i, (v, text) in enumerate(lines):
        raw = HERE / f"line{i}.raw.wav"
        wav = HERE / f"line{i}.wav"
        say(text, v, raw)
        to44(raw, wav)
        audio, sr = sf.read(str(wav), dtype="float32")
        assert sr == SR, sr
        track += [audio, gap]
        voice += [v] * len(audio) + [""] * len(gap)

    speech = np.concatenate(track).astype(np.float32)
    n = len(speech)
    t = np.arange(n) / SR
    # "music": a slow chord + beat, loud enough that demucs has to work for it
    chord = sum(0.05 * np.sin(2 * np.pi * f * t) for f in (110, 165, 220))
    beat = 0.05 * np.sin(2 * np.pi * 55 * t) * (np.sin(2 * np.pi * 2 * t) > 0.7)
    music = (chord + beat).astype(np.float32)
    mix = 0.9 * speech + music
    mix /= max(1.0, float(np.max(np.abs(mix))) / 0.97)

    wav = HERE / "scene.wav"
    sf.write(str(wav), np.stack([mix, mix], axis=1), SR, subtype="PCM_16")

    out = HERE / "test_episode.mkv"
    subprocess.run(
        [ff(), "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=0x223344:s=640x360:r=25",
         "-i", str(wav), "-shortest",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", str(out)],
        check=True,
    )
    info = sf.info(str(wav))
    print(f"wrote {out}  ({info.duration:.1f}s)")
    print("voices:", sorted(set(voice)))
    for f in HERE.glob("line*.wav"):
        f.unlink()


if __name__ == "__main__":
    sys.exit(main())
