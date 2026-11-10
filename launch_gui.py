"""Entry point for the packaged app: the window, or a command-line run with --cli.

  DubTool.exe                       opens the window
  DubTool.exe --cli live x.mkv ...  the same thing the dub.bat commands do
"""

import sys

from dubkit import configure_cache, gui

if __name__ == "__main__":
    configure_cache()
    argv = sys.argv[1:]
    if argv[:1] == ["--cli"]:
        import dub

        sys.argv = ["dub", *argv[1:]]
        raise SystemExit(dub.main(argv[1:]))
    raise SystemExit(gui.main(argv))
