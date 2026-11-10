"""Dub a script, not a video.

Animators and game studios already have their dialogue as separate lines: one WAV per
line, or a CSV of text with a character name per row. There is no mix to separate and
nothing to transcribe, so this mode skips demucs, whisper and diarization entirely.

    dubtool batch lines.csv --lang es --voices cast/
    dubtool batch lines.csv --lang es --voice me.wav --jobs 4

CSV columns are detected by name (case-insensitive): id, character (or speaker/who),
text (or line/dialogue) and optionally audio (a file to match in length).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import ff, voice

AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".aiff", ".aif", ".ogg", ".opus"}

_ID = ("id", "key", "line_id", "lineid", "#", "index")
_WHO = ("character", "speaker", "who", "role", "name", "actor")
_TEXT = ("text", "line", "dialogue", "source", "english", "script")
_AUDIO = ("audio", "file", "wav", "clip", "recording", "path")


@dataclass
class Row:
    id: str
    who: str
    text: str
    audio: Path | None = None


@dataclass
class Config:
    csv: Path
    lang: str
    source_lang: str = "auto"
    out_dir: Path | None = None
    voices: Path | None = None
    voice: Path | None = None
    tts: str = "auto"
    tts_model: str | None = None
    device: str = "auto"
    translate: str = "auto"
    translate_model: str = "google/gemini-2.5-flash"
    jobs: int = 1
    loudnorm: bool = True
    force: bool = False
    warnings: list = field(default_factory=list)


def _pick(header: list[str], names: tuple[str, ...]) -> int | None:
    low = [h.strip().lower() for h in header]
    for want in names:
        if want in low:
            return low.index(want)
    return None


def read_rows(path: Path) -> list[Row]:
    """Read a CSV with or without a header. Blank rows and `#` comments are skipped."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        table = [r for r in csv.reader(fh) if r and not r[0].lstrip().startswith("#")]
    if not table:
        raise SystemExit(f"{path} is empty")
    head = table[0]
    i_id, i_who, i_text, i_audio = _pick(head, _ID), _pick(head, _WHO), _pick(head, _TEXT), _pick(head, _AUDIO)
    has_header = i_text is not None
    rows: list[Row] = []
    for n, rec in enumerate(table[1:] if has_header else table, start=1):
        if i_text is None:  # no header: id, character, text
            rid, who, text = (rec + ["", "", ""])[:3]
            audio = rec[3] if len(rec) > 3 else ""
        else:
            rid = rec[i_id] if i_id is not None and i_id < len(rec) else str(n)
            who = rec[i_who] if i_who is not None and i_who < len(rec) else "narrator"
            text = rec[i_text]
            audio = rec[i_audio] if i_audio is not None and i_audio < len(rec) else ""
        text = text.strip()
        if not text:
            continue
        audio_path = Path(audio.strip()) if audio.strip() else None
        if audio_path and not audio_path.is_absolute():
            audio_path = (path.parent / audio_path).resolve()
        rows.append(Row(id=str(rid).strip() or str(n), who=who.strip() or "narrator",
                        text=text, audio=audio_path))
    if not rows:
        raise SystemExit(f"{path} has no lines")
    return rows


def find_voice(voices: Path, who: str) -> Path | None:
    """`cast/wilbur.wav` answers to Wilbur, WILBUR and "Wilbur Post"."""
    if not voices.is_dir():
        raise SystemExit(f"--voices {voices} is not a folder")
    files = {f.stem.lower(): f for f in voices.iterdir() if f.suffix.lower() in AUDIO_EXT}
    key = who.strip().lower()
    for candidate in (key, key.replace(" ", "_"), key.split()[0], key.replace(".", "")):
        if candidate in files:
            return files[candidate]
    return files.get("default")


def _ref_spk(path: Path | None) -> dict:
    if path is None:
        return {}
    with ff.WavReader(str(path)) as r:
        seconds = r.f.frames / r.sr
    return {"ref": str(path), "ref_seconds": round(seconds, 2)}


def run(cfg: Config) -> dict:
    rows = read_rows(cfg.csv)
    lang = voice.norm_lang(cfg.lang)
    out_dir = cfg.out_dir or cfg.csv.parent / f"{cfg.csv.stem}.{lang}"
    out_dir.mkdir(parents=True, exist_ok=True)

    characters = sorted({r.who for r in rows})
    print(f"  {len(rows)} lines, {len(characters)} characters: {', '.join(characters[:8])}")

    refs: dict[str, Path] = {}
    if cfg.voices:
        for who in characters:
            found = find_voice(cfg.voices, who)
            if found:
                refs[who] = found
            else:
                print(f"    ! no voice file for {who!r} in {cfg.voices} (using the engine default)")
    elif cfg.voice:
        single = Path(cfg.voice)
        if not single.exists():
            raise SystemExit(f"--voice {single} not found")
        refs = {who: single for who in characters}
    if refs:
        print("    voices: " + ", ".join(f"{w} <- {p.name}" for w, p in refs.items()))

    segs = [{"id": n, "text": r.text, "speaker": r.who} for n, r in enumerate(rows)]
    print(f"  translating to {lang} ...")
    lines, provider = voice.translate_segments(segs, cfg.source_lang, lang, cfg.translate,
                                               cfg.translate_model, cfg.csv.stem, {})
    print(f"    via {provider}")

    device = cfg.device
    if device == "auto":
        from .pipeline import _has_cuda
        device = "cuda" if _has_cuda() else "cpu"
    engine = voice.make_engine(cfg.tts, lang, device, cfg.tts_model, jobs=cfg.jobs)
    items = []
    for n, row in enumerate(rows):
        spk = _ref_spk(refs.get(row.who))
        items.append((lines[n].strip() or row.text, spk, lang, out_dir / f"{row.id}.{lang}.wav"))

    engine.prepare({}, out_dir)
    print(f"  voicing {len(items)} lines with {engine.name} ...")
    if hasattr(engine, "say_many"):
        engine.say_many(items)
        for bad in getattr(engine, "failures", []):
            print(f"    ! {bad}")
    else:
        for text, spk, lg, out in items:
            engine.say(text, spk, lg, out)
    engine.close()

    # if a line came with its original recording, fit the dub to that length
    manifest = []
    for n, (row, (_, _, _, out)) in enumerate(zip(rows, items)):
        if not out.exists():
            manifest.append({"id": row.id, "character": row.who, "source": row.text,
                             "translated": lines[n], "file": "", "seconds": 0})
            continue
        if row.audio and row.audio.exists():
            with ff.WavReader(str(row.audio)) as r:
                slot = r.f.frames / r.sr
            ff.fit_audio(str(out), slot, str(out))
        with ff.WavReader(str(out)) as r:
            seconds = r.f.frames / r.sr
        manifest.append({"id": row.id, "character": row.who, "source": row.text,
                         "translated": lines[n], "file": str(out), "seconds": round(seconds, 2)})

    (out_dir / f"lines.{lang}.csv").write_text(
        "id,character,source,translated,file,seconds\n" +
        "".join(f'{m["id"]},{m["character"]},"{m["source"]}","{m["translated"]}",{m["file"]},{m["seconds"]}\n'
                for m in manifest), encoding="utf-8")
    (out_dir / f"lines.{lang}.json").write_text(
        json.dumps({"language": lang, "provider": provider, "engine": engine.name, "lines": manifest},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    done = sum(1 for m in manifest if m["file"])
    print(f"  {done}/{len(manifest)} lines -> {out_dir}")
    return {"dir": str(out_dir), "lines": manifest}
