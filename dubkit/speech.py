"""Who is speaking (diarization) and what are they saying (ASR)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import ff


# ---------------------------------------------------------------- transcriber
def transcribe(vocals: Path, model_size: str, device: str, src_lang: str | None = None):
    """faster-whisper on isolated vocals -> (segments, lang, prob)."""
    from faster_whisper import WhisperModel

    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute)
    raw, info = model.transcribe(
        str(vocals),
        language=src_lang,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 350},
        word_timestamps=True,
        condition_on_previous_text=False,  # avoids loops on long episodes
    )
    segs = []
    for i, s in enumerate(raw):
        text = (s.text or "").strip()
        if not text:
            continue
        segs.append(
            {
                "id": len(segs),
                "start": round(float(s.start), 3),
                "end": round(float(s.end), 3),
                "text": text,
            }
        )
    del model
    return segs, info.language, float(info.language_probability)


# --------------------------------------------------------------- diarization
def diarize(vocals: Path, segments, method: str, hf_token: str | None, device: str):
    """-> (turns, used_method). turns = [{'start','end','speaker'}]"""
    method = (method or "auto").lower()
    if method == "single":
        return [], "single"
    if method == "pyannote" and not hf_token:
        raise SystemExit("pyannote diarization needs a free HF_TOKEN (see README)")
    if method in ("auto", "pyannote") and hf_token:
        try:
            return _pyannote(vocals, hf_token, device), "pyannote"
        except Exception as e:
            if method == "pyannote":
                raise
            print(f"    pyannote unavailable ({type(e).__name__}), falling back to ECAPA")
    if method in ("auto", "ecapa"):
        return _ecapa(vocals, segments)
    raise SystemExit(f"unknown diarization method: {method}")


def _pyannote(vocals: Path, token: str, device: str):
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    pipeline.to(torch.device(device))
    out = pipeline(str(vocals))
    return [
        {"start": round(float(t.start), 3), "end": round(float(t.end), 3), "speaker": spk}
        for t, _, spk in out.itertracks(yield_label=True)
    ]


_ECAPA = None


def _ecapa_encoder():
    global _ECAPA
    if _ECAPA is not None:
        return _ECAPA
    try:  # speechbrain >= 1.0
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:  # older
        from speechbrain.pretrained import EncoderClassifier

    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache")) / "speechbrain"
    cache.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "source": "speechbrain/spkrec-ecapa-voxceleb",
        "savedir": str(cache / "spkrec-ecapa-voxceleb"),
    }
    try:  # Windows users can't create symlinks without developer mode
        from speechbrain.utils.fetching import LocalStrategy

        kwargs["local_strategy"] = LocalStrategy.COPY
    except Exception:
        pass
    try:
        _ECAPA = EncoderClassifier.from_hparams(**kwargs)
    except TypeError:
        kwargs.pop("local_strategy", None)
        _ECAPA = EncoderClassifier.from_hparams(**kwargs)
    return _ECAPA


def _ecapa(vocals: Path, segments):
    """Tokenless diarization: ECAPA embeddings per ASR segment + agglomerative
    clustering. Won't beat pyannote, but needs no HF account."""
    import torch
    from sklearn.cluster import AgglomerativeClustering

    enc = _ecapa_encoder()
    units = [s for s in segments if (s["end"] - s["start"]) >= 0.35]
    if len(units) < 2:
        return [], "single"

    embs = []
    with ff.WavReader(vocals) as r:
        for s in units:
            clip = r.slice(s["start"], s["end"], sr=16000)
            if len(clip) < 1600:
                embs.append(None)
                continue
            with torch.no_grad():
                e = enc.encode_batch(torch.from_numpy(clip).unsqueeze(0)).squeeze().cpu().numpy()
            embs.append(e / (np.linalg.norm(e) + 1e-9))
    idx = [i for i, e in enumerate(embs) if e is not None]
    if len(idx) < 2:
        return [], "single"
    X = np.stack([embs[i] for i in idx])

    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0.62, metric="cosine", linkage="average"
    ).fit_predict(X)
    labels = _merge_similar(X, labels, threshold=0.72)
    labels = _fold_small(X, labels, min_share=0.03)

    turns = []
    for k, i in enumerate(idx):
        u = units[i]
        turns.append({"start": u["start"], "end": u["end"], "speaker": f"spk{labels[k]:02d}"})
    return turns, "ecapa"


def _merge_similar(X, labels, threshold=0.72):
    """Collapse clusters whose centroids are near-identical (over-clustering)."""
    names = sorted(set(labels))
    cent = {n: X[labels == n].mean(axis=0) for n in names}
    parent = {n: n for n in names}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ca, cb = cent[a], cent[b]
            sim = float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-9))
            if sim > threshold:
                parent[find(b)] = find(a)
    return np.array([find(l) for l in labels])


def _fold_small(X, labels, min_share=0.03):
    """Reassign clusters with almost no screen time to their nearest big cluster."""
    names, counts = np.unique(labels, return_counts=True)
    if len(names) < 2:
        return labels
    total = counts.sum()
    big = [n for n, c in zip(names, counts) if c / total >= min_share]
    if not big:
        return np.zeros_like(labels)
    cent = {n: X[labels == n].mean(axis=0) for n in big}
    out = labels.copy()
    for i, l in enumerate(labels):
        if l in big:
            continue
        best = max(big, key=lambda n: float(X[i] @ cent[n]))
        out[i] = best
    remap = {n: i for i, n in enumerate(sorted(set(out)))}
    return np.array([remap[l] for l in out])


def assign_speakers(segments, turns):
    """Attach a speaker to every ASR segment by maximum time overlap."""
    if not turns:
        for s in segments:
            s["speaker"] = "spk00"
        return segments
    prev = "spk00"
    for s in segments:
        best, best_ov = None, 0.0
        for t in turns:
            ov = min(s["end"], t["end"]) - max(s["start"], t["start"])
            if ov > best_ov:
                best, best_ov = t["speaker"], ov
        s["speaker"] = best or prev
        prev = s["speaker"]
    return segments


# ------------------------------------------------------------- voice samples
def build_refs(vocals: Path, segments, refdir: Path, sr: int = 22050,
               min_sec=4.0, max_sec=20.0):
    """One reference clip per speaker, taken from that character's own lines.

    This clip is what the TTS clones, i.e. "the character's original voice".
    """
    refdir = Path(refdir)
    refdir.mkdir(parents=True, exist_ok=True)
    by_spk: dict[str, list] = {}
    for s in segments:
        by_spk.setdefault(s["speaker"], []).append(s)

    info: dict[str, dict] = {}
    with ff.WavReader(vocals) as r:
        for spk, segs in sorted(by_spk.items()):
            segs = sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True)
            picked, total = [], 0.0
            for s in segs:
                dur = s["end"] - s["start"]
                if dur < 0.6:
                    continue
                picked.append(s)
                total += min(dur, 8.0)
                if total >= max_sec:
                    break
            picked.sort(key=lambda s: s["start"])
            chunks = []
            for s in picked:
                clip = r.slice(s["start"], min(s["end"], s["start"] + 8.0), sr=sr)
                if len(clip) < int(0.2 * sr):
                    continue
                chunks.append(clip)
                if sum(len(c) for c in chunks) / sr >= max_sec:
                    break
            audio = (
                np.concatenate([np.concatenate([c, np.zeros(int(0.15 * sr), np.float32)]) for c in chunks])
                if chunks
                else np.zeros(sr, np.float32)
            )
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            if peak > 0:
                audio = audio / peak * 0.9
            path = refdir / f"{spk}.wav"
            ff.write_wav(path, audio, sr)
            speech = sum(s["end"] - s["start"] for s in segs)
            info[spk] = {
                "ref": str(path),
                "ref_seconds": round(len(audio) / sr, 2),
                "segments": len(segs),
                "speech_seconds": round(speech, 2),
                "usable": (len(audio) / sr) >= min_sec,
            }
    return info


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
