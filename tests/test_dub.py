#!/usr/bin/env python
"""Offline checks for the parts that don't need 5 GB of models.

  python tests/test_dub.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubkit import ff, speech, voice  # noqa: E402
from dubkit.pipeline import Work, _sig_file  # noqa: E402


def test_langs():
    assert voice.norm_lang("PT_BR") == "pt"
    assert voice.norm_lang("Chinese") == "zh-cn"
    assert voice.norm_lang("JP") == "ja"
    assert "zh-cn" in voice.XTTS_LANGS
    assert "uk" not in voice.XTTS_LANGS and "uk" in voice.ELEVEN_LANGS


def test_parse_json():
    fenced = 'Sure!\n```json\n[{"id": 1, "text": "hola"}, {"id": 2, "text": "adios"}]\n```\nHope that helps.'
    got = voice.parse_json_array(fenced)
    assert [x["id"] for x in got] == [1, 2]
    assert got[0]["text"] == "hola"
    try:
        voice.parse_json_array("no array here")
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_chunks_cover_everything():
    segs = [{"id": i, "text": "x" * 100} for i in range(75)]
    chunks = voice._chunks(segs, max_chars=350, max_items=30)
    assert [s["id"] for c in chunks for s in c] == list(range(75))
    assert all(len(c) <= 30 for c in chunks)
    assert all(sum(len(s["text"]) for s in c) <= 350 for c in chunks)
    assert all(len(c) > 0 for c in chunks)


def test_assign_speakers():
    segs = [
        {"start": 0.0, "end": 2.0, "text": "a"},
        {"start": 2.1, "end": 4.0, "text": "b"},
        {"start": 4.1, "end": 5.0, "text": "c"},
    ]
    turns = [
        {"start": 0.0, "end": 1.9, "speaker": "spk00"},
        {"start": 2.0, "end": 3.5, "speaker": "spk01"},
        {"start": 5.5, "end": 6.0, "speaker": "spk01"},
    ]
    speech.assign_speakers(segs, turns)
    assert [s["speaker"] for s in segs] == ["spk00", "spk01", "spk01"]  # 3rd inherits
    # no turns at all -> everyone is spk00
    empty = [{"start": 0, "end": 1, "text": "x"}]
    speech.assign_speakers(empty, [])
    assert empty[0]["speaker"] == "spk00"


def test_cluster_merge_and_fold():
    rng = np.random.default_rng(0)

    def unit(v):
        return v / np.linalg.norm(v)

    a = unit(rng.normal(size=8))
    b = unit(rng.normal(size=8))
    X = np.stack([a, unit(a + 0.01 * rng.normal(size=8)), b, unit(b + 0.01 * rng.normal(size=8))])
    merged = speech._merge_similar(X, np.array([0, 1, 2, 3]), threshold=0.9)
    assert len(set(merged)) == 2, merged  # 0+1, 2+3

    # four lines of one character, one stray line -> stray folds into the big cluster
    X2 = np.stack([a, a, a, a, b])
    folded = speech._fold_small(X2, np.array([0, 0, 0, 0, 1]), min_share=0.3)
    assert len(set(folded)) == 1, folded


def test_dub_track_placement():
    sr = 1000
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ff.write_wav(td / "a.wav", np.full(int(0.25 * sr), 0.5, np.float32), sr)
        ff.write_wav(td / "b.wav", np.full(int(0.25 * sr), -0.5, np.float32), sr)
        out = voice.build_dub_track([(0.0, td / "a.wav"), (0.5, td / "b.wav")], sr, 1.0, td / "mix.wav")
        data, got_sr = ff.sf.read(str(out), dtype="float32")
        assert got_sr == sr and len(data) == sr
        assert abs(data[100] - 0.5) < 0.02 and abs(data[600] + 0.5) < 0.02
        assert abs(data[400]) < 0.01 and abs(data[900]) < 0.01

        # a short overlap is allowed: both lines stay on their own timestamp
        out2 = voice.build_dub_track([(0.0, td / "a.wav"), (0.2, td / "b.wav")], sr, 1.0, td / "mix2.wav")
        d2, _ = ff.sf.read(str(out2), dtype="float32")
        assert abs(d2[50] - 0.5) < 0.02 and abs(d2[350] + 0.5) < 0.02

        # a line that runs long is cut at the next one instead of dragging it later
        ff.write_wav(td / "long.wav", np.full(int(2.0 * sr), 0.4, np.float32), sr)
        ff.write_wav(td / "c.wav", np.full(int(0.1 * sr), -0.4, np.float32), sr)
        out3 = voice.build_dub_track(
            [(0.0, td / "long.wav"), (1.0, td / "c.wav")], sr, 3.0, td / "mix3.wav"
        )
        d3, _ = ff.sf.read(str(out3), dtype="float32")
        assert abs(d3[500] - 0.4) < 0.02      # at its own timestamp
        assert abs(d3[1200] - 0.4) < 0.02     # still speaking just before the cut
        assert abs(d3[1290]) < 0.02           # trimmed at 1.25 s, not pushed to 1.0 s
        assert abs(d3[1050]) < 0.03           # 0.4 + (-0.4): the later line kept its slot


def test_fit_audio():
    sr = 22050
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        t = np.arange(int(2.0 * sr)) / sr
        ff.write_wav(td / "in.wav", 0.3 * np.sin(2 * np.pi * 220 * t), sr)
        ff.fit_audio(td / "in.wav", td / "out.wav", target_sec=1.6, ar=44100)
        assert abs(ff.duration(td / "out.wav") - 1.6) < 0.12
        info = ff.sf.info(str(td / "out.wav"))
        assert info.samplerate == 44100 and info.channels == 1
        # extreme fit is clamped, not robot-stretched
        ff.fit_audio(td / "in.wav", td / "out2.wav", target_sec=0.4, ar=22050)
        assert 1.3 < ff.duration(td / "out2.wav") < 1.7


def test_duck_mix():
    """Dialogue goes, laughter between lines stays. (Regression: the gate was inverted once,
    which silenced nothing at all and left the dub playing over the original English.)"""
    sr = 44100
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        t = np.arange(int(2.0 * sr)) / sr
        tone = 0.2 * np.sin(2 * np.pi * 300 * t)
        ff.write_wav(td / "vocals.wav", np.stack([tone, tone], 1), sr)
        ff.write_wav(td / "bed.wav", np.zeros((len(t), 2), dtype="float32"), sr)
        out = ff.duck_mix(td / "bed.wav", td / "vocals.wav", td / "out.wav", [(0.5, 1.0)])
        d, _ = ff.sf.read(str(out), dtype="float32", always_2d=True)

        def lvl(a, b):
            s = d[int(a * sr):int(b * sr)]
            return float(np.sqrt((s ** 2).mean()))

        assert lvl(1.1, 1.9) > 0.1, "sound between lines must survive"
        assert lvl(0.55, 0.95) < 0.005, "the dialogue itself must be gone"
        assert lvl(0.5 - 0.2, 0.5 - 0.13) > 0.1, "the ramp must leave the pause alone"


def test_srt():
    segs = [
        {"start": 0.0, "end": 1.5, "speaker": "spk00", "text": "hi", "text_tr": "hola"},
        {"start": 61.25, "end": 62.0, "speaker": "spk01", "text": "x", "text_tr": ""},
    ]
    with tempfile.TemporaryDirectory() as td:
        p = voice.write_srt(segs, Path(td) / "s.srt")
        body = p.read_text(encoding="utf-8")
        assert "00:00:00,000 --> 00:00:01,500" in body
        assert "hola" in body
        assert "00:01:01,250 --> 00:01:02,000" not in body  # empty line skipped


def test_stage_cache():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        w = Work(td)
        calls = []

        def fn():
            calls.append(1)
            p = td / "art.txt"
            p.write_text("x")
            return [p]

        w.stage("thing", ["v1"], fn)
        Work(td).stage("thing", ["v1"], fn)          # new Work, same state file
        assert len(calls) == 1, "second run should hit cache"
        Work(td).stage("thing", ["v2"], fn)          # settings changed
        assert len(calls) == 2, "changed signature must re-run"
        Work(td, force=True).stage("thing", ["v2"], fn)
        assert len(calls) == 3, "--force must re-run"
        # artifact deleted -> cache invalid
        (td / "art.txt").unlink()
        Work(td).stage("thing", ["v2"], fn)
        assert len(calls) == 4
        assert (td / "state.json").exists()


def test_sig_file():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "f"
        assert _sig_file(p) == "missing"
        p.write_text("a")
        assert _sig_file(p) != "missing"


def test_batch_csv():
    """The script mode's input: a CSV with a header, or bare rows with no header at all."""
    from dubkit import batch

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "lines.csv"
        p.write_text('id,character,text\n001,Wilbur,"Mmm, food."\n# a comment, ignored\n002,Lawyer,"Now then."\n',
                     encoding="utf-8")
        rows = batch.read_rows(p)
        assert [r.id for r in rows] == ["001", "002"], rows
        assert [r.who for r in rows] == ["Wilbur", "Lawyer"]
        assert rows[0].text == "Mmm, food."

        bare = Path(td) / "bare.csv"
        bare.write_text("a,Wilbur,Hi there\nb,Lawyer,Hm.\n", encoding="utf-8")
        rows = batch.read_rows(bare)
        assert [(r.id, r.who, r.text) for r in rows] == [("a", "Wilbur", "Hi there"),
                                                        ("b", "Lawyer", "Hm.")]


def test_batch_voices():
    """cast/wilbur.wav has to answer to Wilbur, WILBUR, wilbur_2 and \"Wilbur Post\"."""
    from dubkit import batch

    with tempfile.TemporaryDirectory() as td:
        cast = Path(td)
        (cast / "wilbur.wav").write_bytes(b"x")
        (cast / "lawyer.mp3").write_bytes(b"x")
        (cast / "default.wav").write_bytes(b"x")
        (cast / "notes.txt").write_bytes(b"x")
        assert batch.find_voice(cast, "Wilbur Post").name == "wilbur.wav"
        assert batch.find_voice(cast, "LAWYER").name == "lawyer.mp3"
        assert batch.find_voice(cast, "Narrator").name == "default.wav"
        (cast / "default.wav").unlink()
        assert batch.find_voice(cast, "Nobody") is None


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  pass  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
