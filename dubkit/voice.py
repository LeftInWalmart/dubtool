"""Translation + voice-cloned speech synthesis + dub track assembly."""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from . import CACHE, ff

# ------------------------------------------------------------------ languages
XTTS_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
    "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
}

# Chatterbox (Resemble AI, MIT) - clones the speaker, 23 languages.
CHATTERBOX_LANGS = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
}

# Piper (MIT, onnxruntime): no voice cloning, but ~1.6x real time on one CPU core, which is the
# only engine here fast enough to dub a stream while somebody watches it. Voices are fixed, so a
# speaker gets their own voice rather than their own voice *copied* - see live.py.
PIPER_VOICES = {
    "es": ["es_MX-claude-high", "es_ES-davefx-medium", "es_AR-daniela-high"],
    "en": ["en_US-lessac-high", "en_GB-cori-high", "en_US-ryan-high"],
    "de": ["de_DE-thorsten-high", "de_DE-kerstin-medium", "de_DE-thorsten_emotional-medium"],
    "fr": ["fr_FR-siwis-medium", "fr_FR-tom-medium", "fr_FR-upmc-medium"],
    "it": ["it_IT-serena-high", "it_IT-paola-medium"],
    "pt": ["pt_BR-cadu-medium", "pt_BR-faber-medium", "pt_BR-jeff-medium"],
    "ru": ["ru_RU-irina-medium", "ru_RU-denis-medium", "ru_RU-dmitri-medium"],
    "zh": ["zh_CN-huayan-medium", "zh_CN-chaowen-medium"],
    "ar": ["ar_JO-kareem-medium"],
    "hi": ["hi_IN-priyamvada-medium", "hi_IN-pratham-medium", "hi_IN-rohan-medium"],
    "ja": ["ja_JP-hi_fi_captain-medium"],
    "ko": ["ko_KR-kss-medium"],
    "nl": ["nl_NL-alex-medium", "nl_BE-nathalie-medium", "nl_NL-mls-medium"],
    "pl": ["pl_PL-gosia-medium", "pl_PL-darkman-medium", "pl_PL-bass-high"],
    "sv": ["sv_SE-alma-medium", "sv_SE-lisa-medium", "sv_SE-nst-medium"],
    "tr": ["tr_TR-dfki-medium"],
    "da": ["da_DK-talesyntese-medium"],
    "no": ["no_NO-talesyntese-medium", "no_NO-nvcc-medium"],
    "fi": ["fi_FI-harri-medium"],
    "el": ["el_GR-joy-medium", "el_GR-rapunzelina-medium"],
    "he": ["he_IL-saspeech-medium"],
    "sw": ["sw_CD-lanfrica-medium"],
    "cs": ["cs_CZ-jirka-medium", "cs_CZ-kasandra-medium"],
    "hu": ["hu_HU-imre-medium", "hu_HU-anna-medium", "hu_HU-berta-medium"],
}
PIPER_LANGS = set(PIPER_VOICES)

# Spoken by at least one engine, so the CLI can say "pick a voice" rather than "unknown".
ALL_LANGS = XTTS_LANGS | CHATTERBOX_LANGS | PIPER_LANGS

ELEVEN_LANGS = {
    "en", "ja", "zh", "de", "hi", "fr", "ko", "pt", "it", "es", "id", "nl",
    "tr", "fil", "pl", "sv", "bg", "ro", "ar", "cs", "el", "fi", "hr", "ms",
    "sk", "da", "ta", "uk", "ru",
}

_ALIASES = {
    "zh": "zh-cn", "cmn": "zh-cn", "zh-hans": "zh-cn", "chinese": "zh-cn",
    "mandarin": "zh-cn", "zh-tw": "zh-cn", "jp": "ja", "jpn": "ja",
    "japanese": "ja", "kr": "ko", "kor": "ko", "korean": "ko",
    "pt-br": "pt", "pt_br": "pt", "brazilian": "pt", "spanish": "es",
    "french": "fr", "german": "de", "italian": "it", "russian": "ru",
    "hindi": "hi", "arabic": "ar", "turkish": "tr", "polish": "pl",
    "dutch": "nl", "czech": "cs", "hungarian": "hu", "english": "en",
}


def norm_lang(code: str) -> str:
    c = (code or "").strip().lower().replace("_", "-")
    return _ALIASES.get(c, c)


def english_name(code: str) -> str:
    c = norm_lang(code)
    names = {
        "en": "English", "es": "Spanish", "fr": "French", "de": "German",
        "it": "Italian", "pt": "Portuguese", "pl": "Polish", "tr": "Turkish",
        "ru": "Russian", "nl": "Dutch", "cs": "Czech", "ar": "Arabic",
        "zh-cn": "Mandarin Chinese", "ja": "Japanese", "hu": "Hungarian",
        "ko": "Korean", "hi": "Hindi", "uk": "Ukrainian", "sv": "Swedish",
        "da": "Danish", "fi": "Finnish", "no": "Norwegian", "el": "Greek",
        "he": "Hebrew", "th": "Thai", "vi": "Vietnamese", "id": "Indonesian",
        "ro": "Romanian", "bg": "Bulgarian", "sk": "Slovak", "hr": "Croatian",
    }
    return names.get(c, c)


# ---------------------------------------------------------------- translation
def _chunks(segs, max_chars: int = 2400, max_items: int = 30):
    out, cur, n = [], [], 0
    for s in segs:
        L = len(s["text"])
        if cur and (n + L > max_chars or len(cur) >= max_items):
            out.append(cur)
            cur, n = [], 0
        cur.append(s)
        n += L
    if cur:
        out.append(cur)
    return out


def parse_json_array(text: str):
    """LLMs wrap JSON in prose/fences. Rip out the array and parse it."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    a, b = t.find("["), t.rfind("]")
    if a < 0 or b <= a:
        raise ValueError("no JSON array in response")
    return json.loads(t[a : b + 1])


def _prompt_batch(batch, src: str, tgt: str, title: str, cast: dict):
    lines = [json.dumps({"id": s["id"], "speaker": s["speaker"], "text": s["text"]}, ensure_ascii=False) for s in batch]
    cast_txt = ", ".join(f"{k}={v}" for k, v in cast.items()) or "(unnamed)"
    return (
        f"You are dubbing the TV show \"{title}\" from {english_name(src)} into {english_name(tgt)}.\n"
        f"Characters: {cast_txt}\n"
        "Translate each line as spoken dialogue for that character.\n"
        "Rules:\n"
        "- Return ONLY a JSON array of objects with keys \"id\" and \"text\". Same ids, same order, one per input line.\n"
        "- Natural, colloquial, idiomatic spoken language. No narration, no notes, never explain.\n"
        "- Keep it about as long as the original so it fits the mouth movements.\n"
        "- Keep proper nouns/names unchanged. Keep interjections natural in the target language.\n"
        "- If a line is untranslatable noise (e.g. \"[music]\"), leave it empty.\n\n"
        "Lines to translate:\n" + "\n".join(lines)
    )


class OpenRouterTranslator:
    def __init__(self, model: str, key: str | None = None):
        from openai import OpenAI

        key = key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("no OPENROUTER_API_KEY set (use --translate google to skip it)")
        self.model = model
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=180)

    def _one(self, batch, src, tgt, title, cast):
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": _prompt_batch(batch, src, tgt, title, cast)}],
            temperature=0.2,
        )
        got = parse_json_array(r.choices[0].message.content or "")
        return {int(x["id"]): (x.get("text") or "") for x in got if "id" in x}

    def translate(self, segs, src, tgt, title, cast=None):
        cast = cast or {}
        out = {}
        for i, batch in enumerate(_chunks(segs), 1):
            for attempt in (batch, batch[: max(1, len(batch) // 2)], None):
                if attempt is None:
                    for s in batch:  # last resort: one at a time, drop what fails
                        try:
                            out.update(self._one([s], src, tgt, title, cast))
                        except Exception as e:
                            print(f"    line {s['id']} untranslated ({type(e).__name__})")
                    break
                try:
                    out.update(self._one(attempt, src, tgt, title, cast))
                    if attempt is not batch:
                        rest = [s for s in batch if s["id"] not in out]
                        if rest:
                            out.update(self._one(rest, src, tgt, title, cast))
                    break
                except Exception as e:
                    print(f"    batch {i} retry ({type(e).__name__}: {str(e)[:120]})")
                    time.sleep(2)
            print(f"    translated {len(out)}/{len(segs)} lines", end="\r")
        print()
        missing = [s for s in segs if s["id"] not in out]
        if missing:
            raise RuntimeError(f"{len(missing)}/{len(segs)} lines could not be translated")
        return out


class GoogleTranslator:
    """No key, no quality. Fallback only - the free gtx endpoint, stdlib only."""

    URL = "https://translate.googleapis.com/translate_a/single"

    def translate(self, segs, src, tgt, title, cast=None, jobs: int = 1):
        tl = (tgt or "en").split("-")[0]
        out: dict = {}
        lock = threading.Lock()
        done = 0

        def one(s):
            nonlocal done
            query = urllib.parse.urlencode(
                {"client": "gtx", "sl": src or "auto", "tl": tl, "dt": "t", "q": s["text"]}
            )
            text = None
            try:
                with urllib.request.urlopen(f"{self.URL}?{query}", timeout=30) as r:
                    parts = json.loads(r.read().decode("utf-8"))[0]
                text = "".join(p[0] for p in parts if p and p[0]).strip()
            except Exception as e:
                print(f"    line {s['id']} untranslated ({type(e).__name__})")
            with lock:
                done += 1
                print(f"    translated {done}/{len(segs)} lines", end="\r")
            if text:
                out[s["id"]] = text

        # the only stage between the transcript and the first dubbed block, so it is the one
        # worth threading; the free endpoint is happy with a handful of parallel requests
        jobs = max(1, min(int(jobs or 1), 8, len(segs) or 1))
        if jobs == 1:
            for s in segs:
                one(s)
                time.sleep(0.15)
        else:
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                list(pool.map(one, segs))
        print()
        return out


def translate_segments(segments, src_lang, tgt_lang, provider: str, model: str,
                       title: str, cast: dict | None = None, jobs: int = 1):
    provider = (provider or "auto").lower()
    if provider == "none":
        return {s["id"]: s["text"] for s in segments}, "none"
    if provider == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("--translate openrouter needs OPENROUTER_API_KEY")
    if provider in ("auto", "openrouter") and os.environ.get("OPENROUTER_API_KEY"):
        try:
            lines = OpenRouterTranslator(model).translate(segments, src_lang, tgt_lang, title, cast)
            return lines, "openrouter"
        except Exception as e:
            if provider == "openrouter":
                raise SystemExit(f"openrouter translation failed: {e}")
            print(f"    openrouter failed ({str(e)[:120]}), falling back to Google Translate")
    t = GoogleTranslator()
    return t.translate(segments, src_lang, tgt_lang, title, cast, jobs=jobs), "google"


# ---------------------------------------------------------------------- voices
def _pick_refs(speakers: dict) -> None:
    """A clone prompt that is silence makes every engine output noise, so drop those and let the
    character borrow the clearest speaker's clip instead of failing."""
    best = None
    for spk in speakers.values():
        ref, peak = spk.get("ref"), 0.0
        if ref and Path(ref).exists():
            try:
                with ff.WavReader(ref) as r:
                    peak = float(np.max(np.abs(r.slice(0.0, r.f.frames / r.sr))))
            except Exception:
                peak = 0.0
        spk["ref_peak"] = round(peak, 3)
        if peak < 1e-3:
            spk["ref"] = None
        elif best is None or spk.get("ref_seconds", 0) > best.get("ref_seconds", 0):
            best = spk
    if best:
        for spk in speakers.values():
            if not spk.get("ref"):
                spk["ref"] = best["ref"]
                spk["ref_borrowed"] = True


class XttsEngine:
    """Coqui XTTS v2: clones a voice from a reference clip, 17 languages.

    Licence: Coqui Public Model License 1.0.0 - non-commercial use only, and Coqui is gone so
    there is no licence to buy. Fine locally, a problem the day you charge for the output.
    """

    name = "xtts"
    noncommercial = True

    def __init__(self, device: str):
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from TTS.api import TTS

        print("      ! XTTS weights are non-commercial (Coqui Public Model License): "
              "fine for tests, not for dubs you sell")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def prepare(self, speakers: dict, workdir: Path):
        _pick_refs(speakers)

    def say(self, text: str, spk: dict, lang: str, out: Path) -> Path:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.tts.tts_to_file(
            text=text,
            speaker_wav=str(spk["ref"]),
            language=lang,
            file_path=str(out),
            split_sentences=True,
        )
        return out

    def close(self):
        pass


class PiperEngine:
    """Piper: MIT, onnxruntime, ~1.6x real time on one core, 24 languages, no cloning.

    Each speaker gets a different stock voice, so a scene still sounds like a scene. That is a
    weaker promise than Chatterbox's copy of the actor, and the reason this engine exists is
    speed: it is the only one that can dub faster than the viewer watches.
    """

    name = "piper"
    noncommercial = False
    parallel = False  # already faster than real time; lapping it buys nothing worth the RAM

    def __init__(self, device: str = "cpu", model: str | None = None):
        if model:  # --tts-model pins one voice for every speaker
            self.pin = model
        else:
            self.pin = None
        self.dir = CACHE / "piper"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._loaded: dict[str, object] = {}
        self._voice_of: dict[str, str] = {}

    # -- voice resolution ---------------------------------------------------
    def _available(self, lang: str) -> list[str]:
        if self.pin:
            return [self.pin]
        voices = PIPER_VOICES.get(lang)
        if not voices:
            raise SystemExit(
                f"piper cannot speak '{lang}'. It has: {', '.join(sorted(PIPER_LANGS))}.\n"
                "Use --tts chatterbox for voice cloning, or --tts elevenlabs for anything else."
            )
        return voices

    def _ensure(self, voice: str) -> None:
        """Download once, into the project cache rather than the current directory."""
        onnx = self.dir / f"{voice}.onnx"
        if onnx.exists():
            return
        from piper.download_voices import download_voice

        print(f"      downloading piper voice {voice} (~60 MB)")
        download_voice(voice, self.dir)

    def _load(self, voice: str):
        if voice in self._loaded:
            return self._loaded[voice]
        from piper import PiperVoice

        self._ensure(voice)
        loaded = PiperVoice.load(self.dir / f"{voice}.onnx")
        self._loaded[voice] = loaded
        return loaded

    def _voice_for(self, spk_id: str, lang: str) -> str:
        key = f"{lang}:{spk_id}"
        if key not in self._voice_of:
            voices = self._available(lang)
            # stable order, so the same character keeps the same voice across runs
            index = sum(map(ord, spk_id)) % len(voices)
            self._voice_of[key] = voices[index]
        return self._voice_of[key]

    # -- engine interface ---------------------------------------------------
    def prepare(self, speakers: dict, workdir: Path) -> None:
        self.speakers = speakers

    def say(self, text: str, spk: dict, lang: str, out: Path) -> Path:
        import soundfile as sf
        from piper import SynthesisConfig

        voice_name = self._voice_for(spk.get("id", ""), lang)
        voice = self._load(voice_name)
        # 0.95 because a straight read always runs long against an edited dub script, and
        # slowing a clip down is more audible than starting a touch brisk.
        config = SynthesisConfig(length_scale=0.95, normalize_audio=True)
        audio = np.concatenate(
            [chunk.audio_float_array for chunk in voice.synthesize(text, syn_config=config)]
        )
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio, voice.config.sample_rate, subtype="PCM_16")
        return out

    def close(self) -> None:
        self._loaded.clear()


class ElevenEngine:
    """ElevenLabs: instant voice cloning + 29 languages (any language XTTS lacks)."""

    name = "elevenlabs"

    def __init__(self, key: str | None = None, model: str = "eleven_multilingual_v2",
                 keep_voices: bool = False):
        import httpx

        key = key or os.environ.get("ELEVENLABS_API_KEY")
        if not key:
            raise SystemExit("--tts elevenlabs needs ELEVENLABS_API_KEY")
        self.model = model
        self.keep = keep_voices
        self.h = httpx.Client(
            base_url="https://api.elevenlabs.io/v1",
            headers={"xi-api-key": key},
            timeout=300,
        )
        self.made: list[str] = []

    def prepare(self, speakers: dict, workdir: Path):
        for spk, info in speakers.items():
            with open(info["ref"], "rb") as f:
                r = self.h.post(
                    "/voices/add",
                    data={"name": f"dub-{spk}-{int(time.time())}", "description": "temporary dub clone"},
                    files={"files": ("ref.wav", f, "audio/wav")},
                )
            r.raise_for_status()
            info["voice_id"] = r.json()["voice_id"]
            self.made.append(info["voice_id"])
            print(f"    cloned {spk} -> {info['voice_id']}")

    def say(self, text: str, spk: dict, lang: str, out: Path) -> Path:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        r = self.h.post(
            f"/text-to-speech/{spk['voice_id']}",
            params={"output_format": "mp3_44100_128"},
            json={
                "text": text,
                "model_id": self.model,
                "voice_settings": {
                    "stability": 0.45,
                    "similarity_boost": 0.85,
                    "style": 0.0,
                    "use_speaker_boost": True,
                },
            },
        )
        r.raise_for_status()
        out.write_bytes(r.content)
        return out

    def close(self):
        if self.keep:
            return
        for v in self.made:
            try:
                self.h.delete(f"/voices/{v}")
            except Exception:
                pass


def _cb_lang(lang: str) -> str:
    """Our codes -> Chatterbox language ids (it says 'zh', not 'zh-cn')."""
    lang = (lang or "en").lower()
    if lang.startswith("zh"):
        return "zh"
    if lang in CHATTERBOX_LANGS:
        return lang
    raise SystemExit(
        f"Chatterbox cannot speak '{lang}'. It supports {len(CHATTERBOX_LANGS)} languages: "
        f"{', '.join(sorted(CHATTERBOX_LANGS))}"
    )


def chatterbox_python() -> Path:
    """The interpreter that has chatterbox installed (we keep it out of the main venv)."""
    root = Path(__file__).resolve().parents[1]
    env = os.environ.get("DUBTOOL_TTS_PYTHON")
    cands = [Path(env)] if env else []
    cands += [root / ".venv-tts" / "Scripts" / "python.exe", root / ".venv-tts" / "bin" / "python"]
    for c in cands:
        if c.exists():
            return c
    raise SystemExit(
        "Chatterbox needs its own environment (its torch pin clashes with ours).\n"
        "Install it once with:  powershell -ExecutionPolicy Bypass -File scripts\\setup_chatterbox.ps1\n"
        "Or point DUBTOOL_TTS_PYTHON at a python that has chatterbox-tts installed."
    )


class _Worker:
    """One chatterbox interpreter. Speaks JSON over stdin/stdout, one line at a time."""

    def __init__(self, py: Path, script: Path, device: str = "cpu", threads: int = 0):
        self.threads = threads
        env = dict(os.environ)
        if threads:
            env["OMP_NUM_THREADS"] = str(threads)
            env["MKL_NUM_THREADS"] = str(threads)
        cmd = [str(py), "-u", str(script), "--device", device]
        if threads:
            cmd += ["--threads", str(threads)]
        self.p = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, env=env,
        )
        hello = self._read()
        if not hello.get("ready"):
            raise SystemExit("chatterbox worker did not start - run scripts\\setup_chatterbox.ps1")
        self.sr = hello.get("sr", 24000)

    def _read(self) -> dict:
        """Next JSON reply, skipping any non-protocol line a library printed to stdout."""
        while True:
            line = self.p.stdout.readline()
            if not line:
                return {}
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    def alive(self) -> bool:
        return self.p.poll() is None

    def ask(self, job: dict) -> dict:
        self.p.stdin.write(json.dumps(job) + "\n")
        self.p.stdin.flush()
        return self._read()

    def close(self):
        if self.alive():
            try:
                self.p.stdin.write('{"stop": 1}\n')
                self.p.stdin.flush()
                self.p.wait(timeout=10)
            except Exception:
                self.p.kill()


class ChatterboxEngine:
    """Resemble AI Chatterbox - MIT weights, 23 languages, no per-minute cost.

    Unlike XTTS it lives in a separate venv, so it runs as a worker process: one model load
    per dub, then a JSON line per speaker turn. `jobs > 1` runs several workers in parallel,
    each pinned to a slice of the cores, which is the only big lever on CPU speed: one worker
    using every core wastes most of them (the sampler is not that parallel), while N workers
    at cores/N each scale close to linearly.
    """

    name = "chatterbox"
    noncommercial = False
    parallel = True

    def __init__(self, device: str = "cpu", exaggeration: float | None = None,
                 cfg_weight: float | None = None, temperature: float | None = None,
                 jobs: int = 1):
        self.py = chatterbox_python()
        self.device = device
        self.jobs = max(1, int(jobs))
        self.kw = {"exaggeration": exaggeration, "cfg_weight": cfg_weight, "temperature": temperature}
        self.workers: list[_Worker] = []
        self.n = 0
        self.worker = Path(__file__).parent / "engines" / "chatterbox_worker.py"

    def _threads(self) -> int:
        # measured: 8 threads on a 12-logical-core laptop beats 4, 6 and 12 — past ~2/3 of the
        # cores the intra-op pool only adds contention. Leave headroom for whatever hosts us.
        cores = os.cpu_count() or 4
        return max(1, (cores * 2 // 3) // self.jobs)

    def _spawn(self, count: int) -> list[_Worker]:
        n = max(1, count)
        threads = self._threads() if n > 1 else 0
        return [_Worker(self.py, self.worker, self.device, threads) for _ in range(n)]

    def prepare(self, speakers: dict, workdir: Path):
        _pick_refs(speakers)
        if not self.workers:
            self.workers = self._spawn(1)
        self.sr = self.workers[0].sr

    def _job(self, text: str, spk: dict, lang: str, out: Path) -> dict:
        self.n += 1
        job = {"id": self.n, "text": text, "lang": _cb_lang(lang), "ref": spk.get("ref"), "out": str(out)}
        job.update({k: v for k, v in self.kw.items() if v is not None})
        return job

    def say(self, text: str, spk: dict, lang: str, out: Path) -> Path:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not self.workers or not all(w.alive() for w in self.workers):
            self.close()
            self.workers = self._spawn(1)
        reply = self.workers[0].ask(self._job(text, spk, lang, out))
        if not reply.get("ok"):
            raise RuntimeError(f"chatterbox failed on line {self.n}: {reply.get('error')}")
        return out

    def say_many(self, items, progress=None) -> None:
        """items: iterable of (text, spk, lang, out). Runs `self.jobs` workers in parallel.

        A line that fails is recorded in `self.failures` and skipped, the way the serial
        path does - one bad line should not throw away a twenty-minute render.
        """
        items = list(items)
        self.failures = []
        if self.jobs == 1 or len(items) < 2:
            for i, (text, spk, lang, out) in enumerate(items, 1):
                try:
                    self.say(text, spk, lang, out)
                except Exception as exc:
                    self.failures.append(f"{i}: {type(exc).__name__}: {exc}")
                if progress:
                    progress(i, len(items))
            return

        self.close()
        self.workers = self._spawn(self.jobs)
        self.sr = self.workers[0].sr
        free: queue.Queue = queue.Queue()
        for w in self.workers:
            free.put(w)
        done = 0
        lock = threading.Lock()

        def run(item):
            nonlocal done
            text, spk, lang, out = item
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            w = free.get()
            try:
                reply = w.ask(self._job(text, spk, lang, out))
                if not reply.get("ok"):
                    with lock:
                        self.failures.append(f"{Path(out).name}: {reply.get('error')}")
            except Exception as exc:
                with lock:
                    self.failures.append(f"{Path(out).name}: {type(exc).__name__}: {exc}")
            finally:
                free.put(w)
            with lock:
                done += 1
                if progress:
                    progress(done, len(items))

        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            for f in [pool.submit(run, it) for it in items]:
                f.result()

    def close(self):
        for w in self.workers:
            try:
                w.close()
            except Exception:
                pass
        self.workers = []


def make_engine(kind: str, lang: str, device: str, model: str | None = None, keep_voices=False,
                jobs: int = 1):
    kind = (kind or "auto").lower()
    if kind == "auto":
        # A dub you sell has to be MIT-clean, so prefer Chatterbox when it is installed.
        try:
            chatterbox_python()
            kind = "chatterbox"
        except SystemExit:
            kind = "xtts" if lang in XTTS_LANGS else "elevenlabs"
    if kind == "chatterbox":
        if lang not in CHATTERBOX_LANGS:
            _cb_lang(lang)  # raises with the language list
        return ChatterboxEngine(device=device, jobs=jobs)
    if kind == "xtts":
        if lang not in XTTS_LANGS:
            raise SystemExit(
                f"XTTS v2 cannot speak '{lang}'. Supported: {', '.join(sorted(XTTS_LANGS))}.\n"
                "Use --tts chatterbox (MIT) or --tts elevenlabs for other languages."
            )
        return XttsEngine(device)
    if kind == "piper":
        if lang not in PIPER_LANGS:
            PiperEngine()._available(lang)  # raises with the language list
        return PiperEngine(device=device, model=model)
    if kind == "elevenlabs":
        return ElevenEngine(model=model or "eleven_multilingual_v2", keep_voices=keep_voices)
    raise SystemExit(f"unknown tts engine: {kind}")


# ----------------------------------------------------------------- dub track
def build_dub_track(items, sr: int, total_sec: float, out: Path, max_overlap: float = 0.25,
                    stats: dict | None = None) -> Path:
    """Lay each dubbed line at its original start time on a silent canvas.

    Every line keeps its own timestamp. A line that runs past the next one is trimmed
    at the next start (40 ms fade) instead of being pushed later, because a dub that
    slides off the picture is worse than one that loses a syllable — and pushing
    later compounds: five long lines put the last one half a minute out of sync.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = int(round(total_sec * sr))
    buf = np.zeros(total, dtype=np.float32)

    clips = []
    for start, path in sorted(items, key=lambda x: x[0]):
        if not Path(path).exists():
            continue
        data, _ = sf.read(str(path), dtype="float32", always_2d=True)
        if len(data) and len(data[0]):
            clips.append((float(start), data[:, 0].copy()))

    for i, (start, data) in enumerate(clips):
        at = min(max(int(round(start * sr)), 0), max(total - 1, 0))
        limit = total
        if i + 1 < len(clips):
            nxt = int(round(clips[i + 1][0] * sr))
            limit = min(total, max(nxt, at + int(0.3 * sr)) + int(max_overlap * sr))
        data = data[: max(0, limit - at)]
        if len(data) == 0:
            continue
        if stats is not None and len(data) < len(clips[i][1]) - int(0.05 * sr):
            stats["cut"] = stats.get("cut", 0) + 1
        fade = min(len(data), int(0.04 * sr))
        data[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        buf[at : at + len(data)] += data

    # Overlapping lines can sum past full scale, and PCM_16 would clip them flat.
    # One global trim keeps the relative levels intact; the mix loudnorms afterwards anyway.
    peak = float(np.abs(buf).max()) if total else 0.0
    if peak > 0.99:
        buf *= 0.99 / peak
        if stats is not None:
            stats["peak_trim"] = round(0.99 / peak, 3)

    sf.write(str(out), buf, sr, subtype="PCM_16")
    return out


def write_srt(segments, out: Path, key: str = "text_tr") -> Path:
    def ts(t):
        ms = int(round(t * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, s in enumerate(segments, 1):
        text = (s[key] if key in s else s.get("text")) or ""
        text = text.strip()
        if not text:
            continue
        lines += [str(i), f"{ts(s['start'])} --> {ts(s['end'])}", text, ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
