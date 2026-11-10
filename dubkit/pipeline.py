"""End-to-end pipeline: analyze -> translate -> re-voice -> mix -> mux."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import ff, speech, voice

SR_MIX = 44100
SR_ASR = 16000


@dataclass
class Config:
    inp: Path
    lang: str = "es"
    out: Path | None = None
    work: Path | None = None
    tts: str = "auto"
    tts_model: str | None = None
    translate: str = "auto"
    translate_model: str = "google/gemini-2.5-flash"
    whisper: str = "large-v3"
    source_lang: str | None = None
    diarize: str = "auto"
    device: str = "auto"
    separate: str = "demucs"
    cast: dict = field(default_factory=dict)
    subtitles: bool = True
    keep_original_audio: bool = False
    keep_voices: bool = True
    loudnorm: bool = True
    jobs: int = 1
    force: bool = False

    def __post_init__(self):
        self.inp = Path(self.inp)
        self.lang = voice.norm_lang(self.lang)
        if self.out is None:
            self.out = self.inp.with_name(f"{self.inp.stem}.{self.lang}.mp4")
        self.out = Path(self.out)
        if self.work is None:
            self.work = self.out.parent / "dubwork" / self.inp.stem
        self.work = Path(self.work)
        if self.device == "auto":
            self.device = "cuda" if _has_cuda() else "cpu"


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _sig_file(p) -> str:
    p = Path(p)
    try:
        return f"{p.stat().st_mtime_ns}"
    except OSError:
        return "missing"


class Work:
    """Per-episode scratch dir with stage-level caching (re-runs are cheap)."""

    def __init__(self, root: Path, force: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.force = force
        self._state_path = self.root / "state.json"
        try:
            self.state = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            self.state = {}

    def save(self):
        self._state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def __truediv__(self, name: str) -> Path:
        return self.p(name)

    def __fspath__(self) -> str:
        return str(self.root)

    def stage(self, name: str, sig: list, fn):
        h = hashlib.sha1(json.dumps(sig, default=str, sort_keys=True).encode()).hexdigest()[:12]
        rec = self.state.get(name)
        if not self.force and rec and rec.get("sig") == h:
            if all(Path(f).exists() for f in rec.get("files", [])):
                print(f"    {name}: cached")
                return
        t0 = time.time()
        print(f"    {name}: running ...")
        files = fn() or []
        self.state[name] = {
            "sig": h,
            "files": [str(f) for f in files],
            "seconds": round(time.time() - t0, 1),
        }
        self.save()
        print(f"    {name}: done in {time.time() - t0:.1f}s")
        return

    def p(self, *parts) -> Path:
        return self.root.joinpath(*parts)


def _separate(work: Work, master: Path, model: str = "htdemucs") -> tuple[Path, Path]:
    outdir = work.p("separated")
    ff.run([sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", model, "-o", str(outdir), str(master)])
    vocals = sorted(outdir.rglob("vocals.wav"), key=lambda p: p.stat().st_mtime)[-1]
    bed = sorted(outdir.rglob("no_vocals.wav"), key=lambda p: p.stat().st_mtime)[-1]
    return vocals, bed


def _silence_like(work: Work, total: float) -> Path:
    out = work.p("bed-silence.wav")
    ff.run(
        [
            ff.ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"anullsrc=r={SR_MIX}:cl=stereo",
            "-t", f"{total:.3f}", "-c:a", "pcm_s16le", str(out),
        ]
    )
    return out


# ----------------------------------------------------------- shared stages
# `analyze` and `run` use the same stage names, so analyzing an episode first
# makes the actual dub skip straight to translation.
def stage_extract(work: Work, cfg: Config) -> Path:
    master = work.p("master.wav")
    work.stage(
        "extract",
        [str(cfg.inp), _sig_file(cfg.inp), SR_MIX],
        lambda: [ff.extract_audio(cfg.inp, master, sr=SR_MIX, channels=2)],
    )
    return master


def stage_split(work: Work, cfg: Config, total: float) -> tuple[Path, Path]:
    """vocals stem (dialogue) + `novocals` stem (whatever the separator called not-dialogue)."""
    master = work.p("master.wav")
    vocals, novocals = work.p("vocals.wav"), work.p("novocals.wav")

    def _split():
        if cfg.separate == "none":
            ff.run([ff.ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(master),
                    "-ac", "2", "-c:a", "pcm_s16le", str(vocals)])
            _silence_like(work, total).replace(novocals)
        else:
            v, b = _separate(work, master)
            ff.run([ff.ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(v),
                    "-ac", "2", "-c:a", "pcm_s16le", str(vocals)])
            ff.run([ff.ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(b),
                    "-ac", "2", "-ar", str(SR_MIX), "-c:a", "pcm_s16le", str(novocals)])
        return [vocals, novocals]

    work.stage("separate", [str(master), _sig_file(master), cfg.separate, "split=2"], _split)
    return vocals, novocals


def stage_bed(work: Work, vocals: Path, novocals: Path, segments: list) -> Path:
    """The bed the dub sits on: separator's bed + the dialogue gaps of the vocals stem."""
    bed = work.p("bed.wav")
    spans = [(s["start"], s["end"]) for s in segments]

    def _bed():
        ff.duck_mix(novocals, vocals, bed, spans)
        return [bed]

    work.stage(
        "bed",
        [str(vocals), _sig_file(vocals), str(novocals), _sig_file(novocals), spans, "bedmix=2"],
        _bed,
    )
    return bed


def stage_asr(work: Work, cfg: Config, vocals: Path) -> dict:
    asr_src, asr_json = work.p("asr_src.wav"), work.p("asr.json")
    work.stage(
        "asr_src",
        [str(vocals), _sig_file(vocals)],
        lambda: [ff.extract_audio(vocals, asr_src, sr=SR_ASR, channels=1)],
    )

    def _asr():
        segs, lang, prob = speech.transcribe(asr_src, cfg.whisper, cfg.device, cfg.source_lang)
        speech.dump_json(asr_json, {"lang": lang, "lang_prob": round(prob, 4), "segments": segs})
        print(f"      {len(segs)} lines, detected {lang} ({prob:.0%})")
        return [asr_json]

    work.stage("asr", [str(asr_src), _sig_file(asr_src), cfg.whisper, cfg.device, cfg.source_lang], _asr)
    return speech.load_json(asr_json)


def stage_speakers(work: Work, cfg: Config, vocals: Path, segments) -> dict:
    asr_json, spk_json = work.p("asr.json"), work.p("speakers.json")

    def _spk():
        turns, method = speech.diarize(vocals, segments, cfg.diarize, _hf_token(cfg), cfg.device)
        speech.assign_speakers(segments, turns)
        if len({s["speaker"] for s in segments}) == 1:
            method = "single"
        refs = speech.build_refs(vocals, segments, work.p("refs"))
        speech.dump_json(spk_json, {"method": method, "speakers": refs, "segments": segments, "turns": turns})
        print(f"      {len(refs)} speaker(s) via {method}: " + ", ".join(refs))
        return [spk_json]

    work.stage(
        "speakers",
        [str(asr_json), _sig_file(asr_json), cfg.diarize, cfg.device, bool(_hf_token(cfg))],
        _spk,
    )
    spk = speech.load_json(spk_json)
    for name, info in spk["speakers"].items():
        if not info.get("usable"):
            print(f"      ! {name} has only {info['ref_seconds']}s of clean audio — voice clone may drift")
    return spk


def run(cfg: Config) -> Path:
    t_all = time.time()
    work = Work(cfg.work, force=cfg.force)
    total = ff.duration(cfg.inp)
    print(f"\n=== {cfg.inp.name}  ({total/60:.1f} min)  ->  {voice.english_name(cfg.lang)} ===")

    master = stage_extract(work, cfg)
    vocals, novocals = stage_split(work, cfg, total)
    asr = stage_asr(work, cfg, vocals)
    segments, src_lang = asr["segments"], asr["lang"]
    if not segments:
        raise SystemExit("no speech detected — is the audio track empty?")

    spk = stage_speakers(work, cfg, vocals, segments)
    segments, speakers = spk["segments"], spk["speakers"]
    bed = stage_bed(work, vocals, novocals, segments)

    # 5. translate
    spk_json = work.p("speakers.json")
    tr_json = work.p(f"translate.{cfg.lang}.json")

    def _tr():
        lines, provider = voice.translate_segments(
            segments, src_lang, cfg.lang, cfg.translate, cfg.translate_model,
            cfg.inp.stem, cfg.cast,
        )
        for s in segments:
            s["text_tr"] = lines.get(s["id"], s["text"])
        speech.dump_json(
            tr_json, {"lang": cfg.lang, "provider": provider, "segments": segments}
        )
        return [tr_json]

    work.stage(
        "translate",
        [str(spk_json), _sig_file(spk_json), cfg.lang, cfg.translate, cfg.translate_model, cfg.cast],
        _tr,
    )
    segments = speech.load_json(tr_json)["segments"]

    # 6. speak it in each character's own voice
    dubdir = work.p("dub")
    dub_wavs: list[tuple[float, Path]] = []
    engine_name = cfg.tts

    def _tts():
        nonlocal engine_name
        engine = voice.make_engine(cfg.tts, cfg.lang, cfg.device, cfg.tts_model, cfg.keep_voices,
                                   jobs=cfg.jobs)
        engine_name = engine.name
        try:
            engine.prepare(speakers, work)
            t0 = time.time()
            todo = []  # (segment, speaker info, text, raw, fitted)
            for i, s in enumerate(segments):
                text = (s.get("text_tr") or "").strip()
                if not text:
                    continue
                info = speakers.get(s["speaker"]) or next(iter(speakers.values()))
                todo.append((s, info, text, dubdir / f"{i:05d}.raw.wav", dubdir / f"{i:05d}.wav"))

            def progress(n, total):
                print(f"      {n}/{total} lines  ({time.time() - t0:.0f}s)", end="\r", flush=True)

            if hasattr(engine, "say_many"):
                engine.say_many([(t, info, cfg.lang, raw) for _, info, t, raw, _ in todo], progress)
            else:
                for n, (_, info, t, raw, _) in enumerate(todo, 1):
                    try:
                        engine.say(t, info, cfg.lang, raw)
                    except Exception as e:
                        print(f"      ! line failed ({type(e).__name__}: {e})")
                    progress(n, len(todo))

            for bad in getattr(engine, "failures", []):
                print(f"      ! {bad}")

            items = []
            for s, _info, _t, raw, fitted in todo:
                if not raw.exists() or raw.stat().st_size < 512:
                    continue
                ff.fit_audio(raw, fitted, target_sec=max(0.2, s["end"] - s["start"]), ar=SR_MIX)
                items.append((s["start"], fitted))
            print(f"\n      {len(items)} dubbed lines in {time.time() - t0:.0f}s")
            tag = work.p("tts_engine.txt")
            tag.write_text(engine.name, encoding="utf-8")
            return [tag] + [p for _, p in items]
        finally:
            engine.close()

    work.stage(
        "tts",
        [str(tr_json), _sig_file(tr_json), cfg.tts, cfg.tts_model, cfg.lang, cfg.device, "fit=1.50", "mixer=2"],
        _tts,
    )
    dub_wavs = [
        (s["start"], dubdir / f"{i:05d}.wav")
        for i, s in enumerate(segments)
        if (dubdir / f"{i:05d}.wav").exists()
    ]
    tag = work.p("tts_engine.txt")
    engine_name = tag.read_text(encoding="utf-8").strip() if tag.exists() else cfg.tts

    # 7. lay lines on the timeline, mix under the original bed
    dub_track = work.p("dub_track.wav")
    mixed = work.p("mixed.wav")

    def _mix():
        stats: dict = {}
        voice.build_dub_track(dub_wavs, SR_MIX, total, dub_track, stats=stats)
        if stats.get("cut"):
            print(
                f"    {stats['cut']}/{len(dub_wavs)} lines ran long and were cut at the next cue "
                "— shorten those translations for a clean pass",
                flush=True,
            )
        ff.mix(bed, dub_track, mixed, loudnorm=cfg.loudnorm)
        return [dub_track, mixed]

    work.stage(
        "mix",
        [str(bed), _sig_file(bed), [(str(p), _sig_file(p)) for _, p in dub_wavs], total, cfg.loudnorm, "mixer=4"],
        _mix,
    )

    # 8. subtitles + final video
    srt = None
    if cfg.subtitles:
        srt = voice.write_srt(segments, work.p(f"{cfg.inp.stem}.{cfg.lang}.srt"))

    extras = [master] if cfg.keep_original_audio else []
    final = ff.mux(cfg.inp, mixed, cfg.out, extra_audio=extras, subtitle=srt)

    # 9. report
    report = {
        "input": str(cfg.inp),
        "output": str(cfg.out),
        "video_seconds": round(total, 2),
        "source_language": src_lang,
        "source_confidence": asr.get("lang_prob"),
        "target_language": cfg.lang,
        "speakers": {
            k: {
                "voices_lines": v["segments"],
                "speech_seconds": v["speech_seconds"],
                "reference_clip": v["ref"],
                "usable": v["usable"],
            }
            for k, v in speakers.items()
        },
        "diarization": spk.get("method"),
        "segments": len(segments),
        "dubbed_segments": len(dub_wavs),
        "translator": speech.load_json(tr_json)["provider"],
        "translation_model": cfg.translate_model,
        "tts_engine": engine_name,
        "whisper_model": cfg.whisper,
        "device": cfg.device,
        "elapsed_seconds": round(time.time() - t_all, 1),
        "lines": [
            {
                "id": s["id"],
                "start": s["start"],
                "end": s["end"],
                "speaker": s["speaker"],
                "original": s["text"],
                "dubbed": s.get("text_tr", ""),
            }
            for s in segments
        ],
    }
    (work / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  video : {cfg.out}")
    if srt:
        print(f"  subs  : {srt}")
    print(f"  report: {work / 'report.json'}")
    print(f"  total : {time.time() - t_all:.0f}s\n")
    return cfg.out


def _hf_token(cfg: Config):
    import os

    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


# --------------------------------------------------------------------- analyze
def analyze(cfg: Config) -> dict:
    """Stop before translation: report speakers, voices, transcript."""
    work = Work(cfg.work, force=cfg.force)
    total = ff.duration(cfg.inp)
    stage_extract(work, cfg)
    vocals, _bed = stage_split(work, cfg, total)
    asr = stage_asr(work, cfg, vocals)
    segments = asr["segments"]
    spk = stage_speakers(work, cfg, vocals, segments)
    segments = spk["segments"]
    print(f"\n  {cfg.inp.name}")
    print(f"  duration   : {total/60:.1f} min")
    print(f"  language   : {asr['lang']} ({asr['lang_prob']:.0%})")
    print(f"  lines      : {len(segments)}")
    print(f"  diarization: {spk['method']}")
    for k, v in sorted(spk["speakers"].items(), key=lambda kv: -kv[1]["speech_seconds"]):
        print(f"    {k}: {v['segments']:4d} lines, {v['speech_seconds']/60:5.1f} min speech, ref {v['ref_seconds']}s")
    report = {
        "input": str(cfg.inp),
        "video_seconds": round(total, 2),
        "language": asr["lang"],
        "language_confidence": asr.get("lang_prob"),
        "lines": len(segments),
        "speakers": spk["speakers"],
        "segments": segments,
    }
    path = work / "analyze.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  saved: {path}\n")
    return report
