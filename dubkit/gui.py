"""The window. Everything here is tkinter: no dependencies, no installer, no console.

  dubtool                     (or DubTool.exe)  -> this window
  dubtool gui --shot out.png  -> build it, save a picture of it, exit (used by the tests)

The window is deliberately thin. It builds the same arguments the command line takes and
hands them to `dub.main()`, so there is one dubbing engine and one place where behaviour
lives. The log pane is not decoration: dubbing is slow and silent, and a frozen window is
what makes people think it crashed.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, ttk

from . import __version__, configure_cache, voice

INK = "#0a0a0c"
PANEL = "#121215"
LINE = "#24242a"
CREAM = "#f2ede4"
MUTED = "#8d8a84"
ACCENT = "#ff9d5c"

# codes -> what a person calls them
NAMES = {
    "ar": "Arabic", "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fi": "Finnish", "fr": "French", "he": "Hebrew",
    "hi": "Hindi", "hu": "Hungarian", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "ms": "Malay", "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
    "ru": "Russian", "sv": "Swedish", "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}
WANT = ["es", "de", "fr", "it", "pt", "ja", "ko", "zh", "hi", "ru", "ar", "pl", "nl",
        "tr", "sv", "da", "fi", "no", "cs", "hu", "el", "he", "ms", "sw", "en"]
LANGS = [c for c in WANT if c in voice.ALL_LANGS] + sorted(voice.ALL_LANGS - set(WANT))
LABEL = {c: (NAMES.get(c, c) if c in WANT else c) for c in LANGS}


class Stdout:
    """Anything printed while a job runs lands in the log pane instead of a console."""

    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, text: str) -> int:
        if text:
            self.q.put(text)
        return len(text)

    def flush(self) -> None:
        pass


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.log_q: queue.Queue[str] = queue.Queue()
        self.live = None            # the running Live object, so Stop can reach it
        self.busy = False
        self.out_dir = Path.cwd()

        root.title(f"DubTool {__version__}")
        root.configure(bg=INK)
        root.minsize(760, 560)
        root.geometry("880x620")

        self._style()
        pad = {"padx": 22}

        header = tk.Frame(root, bg=INK)
        header.pack(fill="x", pady=(20, 6), **pad)
        tk.Label(header, text="DubTool", bg=INK, fg=CREAM,
                 font=("Segoe UI", 22, "bold")).pack(side="left")
        tk.Label(header, text="  voice-matched dubbing, on this machine", bg=INK,
                 fg=MUTED, font=("Segoe UI", 10)).pack(side="left", pady=(9, 0))

        # --- source
        src = tk.Frame(root, bg=INK)
        src.pack(fill="x", pady=(10, 0), **pad)
        tk.Label(src, text="VIDEO", bg=INK, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w")
        row = tk.Frame(src, bg=INK)
        row.pack(fill="x", pady=(2, 0))
        self.src = tk.Entry(row, bg=PANEL, fg=CREAM, insertbackground=CREAM, relief="flat",
                            font=("Consolas", 10))
        self.src.pack(side="left", fill="x", expand=True, ipady=7, padx=(0, 8))
        self._button(row, "Choose file…", self.choose, width=13).pack(side="left")
        self.hint = tk.Label(src, text="a video file, or a link (YouTube and most sites, via yt-dlp)",
                             bg=INK, fg=MUTED, font=("Segoe UI", 8))
        self.hint.pack(anchor="w", pady=(4, 0))

        # --- options
        opts = tk.Frame(root, bg=INK)
        opts.pack(fill="x", pady=(16, 0), **pad)
        self.lang = self._combo(opts, "LANGUAGE", [LABEL[c] for c in LANGS], 2)
        self.engine = self._combo(opts, "VOICE ENGINE", ["piper — fast, plain voice",
                                                        "chatterbox — matches the actor, slow"], 2)
        self.whisper = self._combo(opts, "TRANSCRIPTION", ["tiny — fastest", "small — better",
                                                           "medium — best, slow"], 2)
        self.engine.set("piper — fast, plain voice")
        self.whisper.set("small — better")
        self.lang.set(LABEL["es"])

        # --- actions
        actions = tk.Frame(root, bg=INK)
        actions.pack(fill="x", pady=(18, 0), **pad)
        self.play_btn = self._button(actions, "▶  Watch it dub", self.start_live, accent=True)
        self.play_btn.pack(side="left", ipadx=10)
        self.file_btn = self._button(actions, "Save a dubbed file", self.start_file)
        self.file_btn.pack(side="left", padx=(10, 0), ipadx=10)
        self.stop_btn = self._button(actions, "Stop", self.stop, width=8)
        self.stop_btn.pack(side="left", padx=(10, 0))
        self.open_btn = self._button(actions, "Open folder", self.open_folder, width=12)
        self.open_btn.pack(side="right")

        # --- log
        body = tk.Frame(root, bg=INK)
        body.pack(fill="both", expand=True, pady=(14, 0), **pad)
        self.text = tk.Text(body, bg=PANEL, fg="#c9c5bd", insertbackground=CREAM, relief="flat",
                            font=("Consolas", 9), wrap="word", padx=12, pady=10, height=12)
        bar = ttk.Scrollbar(body, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        self.text.tag_configure("accent", foreground=ACCENT)
        self.text.configure(state="disabled")

        self.status = tk.Label(root, text="ready", bg=INK, fg=MUTED, font=("Segoe UI", 9),
                               anchor="w")
        self.status.pack(fill="x", pady=(8, 14), **pad)

        self.write(f"DubTool {__version__}\n", "accent")
        self.write("Pick a video, pick a language, then either watch it while it dubs or save a file.\n")
        self.root.after(80, self.drain)

    # ------------------------------------------------------------------ bits
    def _style(self):
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=CREAM,
                     arrowcolor=CREAM, bordercolor=LINE, lightcolor=PANEL, darkcolor=PANEL,
                     padding=6)
        st.map("TCombobox", fieldbackground=[("readonly", PANEL)],
               foreground=[("readonly", CREAM)])
        st.configure("Vertical.TScrollbar", background=PANEL, troughcolor=INK,
                     bordercolor=INK, arrowcolor=MUTED)

    def _button(self, parent, text, command, accent=False, width=None):
        return tk.Button(parent, text=text, command=command, relief="flat", borderwidth=0,
                         bg=ACCENT if accent else PANEL, fg=INK if accent else CREAM,
                         activebackground="#ffb37e" if accent else LINE,
                         activeforeground=INK if accent else CREAM,
                         font=("Segoe UI", 10, "bold" if accent else "normal"),
                         cursor="hand2", padx=14, pady=8, width=width)

    def _combo(self, parent, label, values, width):
        box = tk.Frame(parent, bg=INK)
        box.pack(side="left", fill="x", expand=True, padx=(0, 14))
        tk.Label(box, text=label, bg=INK, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w")
        cb = ttk.Combobox(box, values=values, state="readonly", width=width,
                          font=("Segoe UI", 10))
        cb.pack(fill="x", pady=(2, 0))
        return cb

    def write(self, text: str, tag: str | None = None):
        self.text.configure(state="normal")
        self.text.insert("end", text, tag or ())
        self.text.see("end")
        self.text.configure(state="disabled")

    def drain(self):
        try:
            while True:
                chunk = self.log_q.get_nowait()
                self.write(chunk)
        except queue.Empty:
            pass
        self.root.after(80, self.drain)

    def choose(self):
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Video", "*.mkv *.mp4 *.m4v *.avi *.mov *.webm *.ts *.mpg *.mpeg *.wmv"),
                       ("All files", "*.*")])
        if path:
            self.src.delete(0, "end")
            self.src.insert(0, path)
            self.out_dir = Path(path).parent

    def open_folder(self):
        target = self.out_dir if self.out_dir.exists() else Path.cwd()
        try:
            os.startfile(str(target))       # windows
        except AttributeError:
            print(f"  {target}")

    # ------------------------------------------------------------------ jobs
    def _code(self, cb, mapping):
        return mapping[cb.get()]

    def _lang(self) -> str:
        return LANGS[[LABEL[c] for c in LANGS].index(self.lang.get())]

    def _engine(self) -> str:
        return "chatterbox" if self.engine.get().startswith("chatterbox") else "piper"

    def _whisper(self) -> str:
        return self.whisper.get().split(" ")[0]

    def _guard(self) -> bool:
        src = self.src.get().strip().strip('"')
        if self.busy:
            self.status.configure(text="already running — press Stop first")
            return False
        if not src:
            self.status.configure(text="choose a video first")
            return False
        if not ("://" in src or Path(src).exists()):
            self.status.configure(text=f"no such file: {src}")
            return False
        return True

    def _run_in_thread(self, fn):
        q = self.log_q
        old = sys.stdout
        sys.stdout = Stdout(q)              # type: ignore[assignment]

        def body():
            try:
                fn()
            except SystemExit as exc:
                q.put(f"\n!! {exc}\n")
            except Exception as exc:
                q.put(f"\n!! {type(exc).__name__}: {exc}\n{traceback.format_exc()}\n")
            finally:
                sys.stdout = old
                self.root.after(0, self._done)

        threading.Thread(target=body, daemon=True).start()
        self.busy = True
        self.play_btn.configure(state="disabled")
        self.file_btn.configure(state="disabled")

    def _done(self):
        self.busy = False
        self.live = None
        self.play_btn.configure(state="normal")
        self.file_btn.configure(state="normal")
        self.status.configure(text="ready")

    def start_live(self):
        if not self._guard():
            return
        src = self.src.get().strip().strip('"')
        lang, engine, whisper = self._lang(), self._engine(), self._whisper()
        self.write(f"\n--- watch it dub: {Path(src).name or src} -> {lang} ({engine})\n", "accent")
        self.status.configure(text="buffering the first blocks — the player window opens by itself")

        from . import live

        cfg = live.LiveConfig(inp=src, lang=lang, tts=engine, whisper=whisper,
                              open_browser=True)
        if engine != "piper":
            self.write("      ! only piper keeps up with playback; a cloning engine will stall\n")

        def job():
            self.live = live.Live(cfg)
            live.Handler.live = self.live
            live.serve(cfg, _live=self.live)

        def wrapped():
            self._run_in_thread(job)

        self._run_in_thread(job)
        self.status.configure(text="running — the player window is open")

    def start_file(self):
        if not self._guard():
            return
        src = self.src.get().strip().strip('"')
        lang, engine, whisper = self._lang(), self._engine(), self._whisper()
        self.write(f"\n--- saving a dubbed file: {Path(src).name or src} -> {lang} ({engine})\n",
                   "accent")
        self.status.configure(text="dubbing — this takes a while, the log shows progress")

        def job():
            import dub
            from . import live

            path = src
            if "://" in src:            # a link has to be fetched before it can be dubbed
                self.log_q.put("  fetching the video…\n")
                path = str(live.resolve_source(src, Path("dubwork") / "downloads"))
            self.out_dir = Path(path).parent
            dub.main(["run", path, "--lang", lang, "--tts", engine, "--whisper", whisper])

        self._run_in_thread(job)

    def stop(self):
        if self.live is not None:
            self.live.shutdown()
            self.write("\n--- stopped\n", "accent")
        else:
            self.write("\n--- nothing to stop (the file dubber finishes or fails)\n")


def self_test(out: Path) -> int:
    """Prove the packaged app can reach its parts: no window, one json file.

    A .exe that imports lazily can look fine and then fail at the first click, so the
    frozen build is checked this way before it is shipped."""
    import importlib
    import json

    report: dict[str, str] = {}
    for name in ("dub", "dubkit.live", "dubkit.stream", "dubkit.voice", "dubkit.speech",
                 "dubkit.pipeline", "dubkit.batch"):
        try:
            importlib.import_module(name)
            report[name] = "ok"
        except Exception as exc:
            report[name] = f"{type(exc).__name__}: {exc}"
    try:
        from . import ff

        report["ffmpeg"] = ff.ffmpeg()
    except Exception as exc:
        report["ffmpeg"] = f"!! {type(exc).__name__}: {exc}"
    try:
        import piper

        report["piper"] = "ok"
    except Exception as exc:
        report["piper"] = f"{type(exc).__name__}: {exc}"
    try:
        import faster_whisper  # noqa: F401

        report["faster_whisper"] = "ok"
    except Exception as exc:
        report["faster_whisper"] = f"{type(exc).__name__}: {exc}"
    try:                                  # the player page must travel inside the bundle
        import threading
        import urllib.request
        from http.server import ThreadingHTTPServer

        from . import live

        live.Handler.live = live.Live(live.LiveConfig(inp="x", lang="es"))
        server = ThreadingHTTPServer(("127.0.0.1", 0), live.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        page = urllib.request.urlopen(url, timeout=10).read().decode()
        report["player.html"] = "ok" if "AudioContext" in page and "DubTool" in page else "wrong page"
        urllib.request.urlopen(url + "state.json", timeout=10).read()
        report["state.json"] = "ok"
        server.shutdown()
    except Exception as exc:
        report["player.html"] = f"!! {type(exc).__name__}: {exc}"
    report["languages"] = str(len(voice.ALL_LANGS))
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    configure_cache()
    if "--self-test" in argv:
        return self_test(Path(argv[argv.index("--self-test") + 1]))
    root = tk.Tk()
    app = App(root)
    shot = None
    if "--shot" in argv:
        shot = Path(argv[argv.index("--shot") + 1])
    if shot:
        def grab():
            root.update_idletasks()
            root.update()
            from PIL import ImageGrab
            x, y = root.winfo_rootx(), root.winfo_rooty()
            w, h = root.winfo_width(), root.winfo_height()
            ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(str(shot))
            print(f"  saved {shot}")
            root.destroy()
        root.after(700, grab)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
