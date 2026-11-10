"""dubkit - analyze a TV episode, then re-voice it in another language
using each character's own voice (speaker-cloned TTS)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "0.3.0"

#: model caches live next to the tool (override with DUB_CACHE)
CACHE = Path(__file__).resolve().parent.parent / ".cache"


def configure_cache() -> Path:
    """Point HF/torch caches at this folder instead of a possibly-full C: drive.

    Must run before any model import. Respects already-set env vars.
    """
    if os.environ.get("DUB_CACHE"):
        root = Path(os.environ["DUB_CACHE"])
    elif getattr(sys, "frozen", False):
        # a packaged app runs from wherever the user dropped it: do not grow a cache next to
        # it, and do not ask for a 2 GB model download every time it is moved
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "DubTool" / "cache"
        root = base
    else:
        root = Path(CACHE)
    subs = (
        ("HF_HOME", "huggingface"),
        ("TORCH_HOME", "torch"),
        ("XDG_CACHE_HOME", "xdg"),
        ("TTS_HOME", "tts"),  # coqui defaults to %LOCALAPPDATA%\tts otherwise
    )
    for key, sub in subs:
        if not os.environ.get(key):
            os.environ[key] = str(root / sub)
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    return root
