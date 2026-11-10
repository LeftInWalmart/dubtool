# Third-party licences

DubTool itself is MIT (see `LICENSE`). The models it downloads are not covered by that
licence, and one of them restricts what you may do with its output:

| model | used for | licence | commercial dubs? |
|---|---|---|---|
| [chatterbox](https://github.com/resemble-ai/chatterbox) | default voice engine | MIT | yes |
| [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) | optional `--tts xtts` | Coqui Public Model License | **no — non-commercial only** |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | transcription | MIT | n/a |
| [demucs](https://github.com/facebookresearch/demucs) | voice/music separation | MIT | n/a |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | speaker diarization | MIT (+ model terms) | n/a |
| [speechbrain](https://github.com/speechbrain/speechbrain) | ECAPA embeddings | Apache-2.0 | n/a |

XTTS-v2 is opt-in and never the default: its licence allows "only non-commercial use of a
machine learning model and its outputs", and Coqui no longer exists to sell you another one.
Fine-tuning on its output does not change that.

Separately: this licence covers the software, not the shows. Dubbing someone else's work needs
their permission and, for a cloned voice, the speaker's.
