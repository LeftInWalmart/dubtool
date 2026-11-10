One window, one button.

**`DubTool.exe` is below.** Download, double-click, nothing to install: Python, ffmpeg and
the transcription model are inside it (231 MB). First run downloads ~1 GB of voice models to
`%LOCALAPPDATA%\DubTool\cache`; after that it works offline.

- pick a video, or paste a link (YouTube and most sites, via yt-dlp)
- pick a language from 26
- **Watch it dub** — a player window opens and starts ~10 s later, while the rest of the film
  is still being dubbed behind it
- **Save a dubbed file** — mp4 with the dub on track 1 and the original on track 2

What this build does not do: match the *actor's* voice. That needs the cloning engine
(chatterbox), which pulls in torch and would triple the download; the app uses piper instead,
which is plain but fast enough to keep up with playback. Clone-quality dubbing is in the
source build: `dub.bat run show.mkv --lang es --tts chatterbox`.

Measured on a Ryzen 5 5600G, no GPU: first frame at 10–13 s for a 60 s clip, the whole clip
dubbed at ~1.5x real time; a 40-minute episode starts just as fast.

Requires Windows 10/11 (uses the Edge/WebView2 runtime that is already on the machine for the
player window).
