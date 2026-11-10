# Contributing

Bug reports with a log are worth more than patches, because the failure modes here are
almost always a specific file: a mix where the laugh track swallows the dialogue, a
translation that runs 2× the slot, a two-speaker scene the diarizer reads as six people.
Attach `dubwork/<name>/report.json` and the first 50 lines of the run.

## Developing

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1          # .venv + deps
powershell -ExecutionPolicy Bypass -File scripts\setup_chatterbox.ps1   # voice engine
.\.venv\Scripts\python.exe tests\test_dub.py                # offline suite, ~2 s
```

The offline suite deliberately needs only `numpy soundfile imageio-ffmpeg`: torch,
whisper and demucs are imported lazily so that CI (and you) can iterate in seconds
without downloading 10 GB.

To hear the result without hunting for footage:

```powershell
.\.venv\Scripts\python.exe dub.py run demo\excerpt.mp4 --lang es --whisper tiny
```

## Rules of the road

- **Never make a non-commercial model the default.** XTTS-v2's weights forbid commercial
  use of their output; Chatterbox (MIT) is the default for that reason. See
  `THIRD-PARTY-LICENSES.md`.
- **Correctness over cleverness in the mix.** A dub that drifts is worse than a dub with a
  clipped line, which is why lines are trimmed to their slot instead of being pushed later.
- **Say what was measured.** If you change a timing, a language count or a licence claim,
  put the number you actually observed in the README, and say on what machine.
- Keep `tests/test_dub.py` runnable in two seconds with no model downloads.
