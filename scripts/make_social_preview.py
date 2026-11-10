"""Render social-preview.png (1200x630) for the GitHub repo card.

GitHub has no API for the social preview — it is a drag-and-drop in
Settings → General → Social preview — so this script exists to make the asset
regenerable rather than to upload it. Needs Pillow and a Windows box with
Georgia/Segoe/Consolas; on other machines point FONT at anything local.

    .venv\\Scripts\\python.exe scripts\\make_social_preview.py
"""

from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1200, 630  # GitHub crops the card to 1.91:1
INK, CREAM, MUTED, ACCENT = (8, 8, 10), (244, 239, 231), (154, 147, 139), (255, 157, 92)

FONT = "C:/Windows/Fonts/"
SERIF, SERIF_I = FONT + "georgia.ttf", FONT + "georgiai.ttf"
SEMIBOLD, REGULAR, MONO = FONT + "seguisb.ttf", FONT + "segoeui.ttf", FONT + "consola.ttf"

ROOT = pathlib.Path(__file__).resolve().parents[1]
X, COL_W = 76, 520  # text column stops short of the still


def fit(draw, font_path, text, size, width, spacing=""):
    """Largest point size at which `text` fits `width`. Hand-spacing defeats wrapping."""
    while size > 10:
        font = ImageFont.truetype(font_path, size)
        rendered = spacing.join(text) if spacing else text
        if draw.textlength(rendered, font=font) <= width:
            return font, rendered
        size -= 1
    return ImageFont.truetype(font_path, 10), text


def main() -> None:
    img = Image.new("RGB", (W, H), INK)

    # Two warm blobs, blurred hard, so the corner reads as light rather than a shape.
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    g = ImageDraw.Draw(glow)
    g.ellipse((820, -300, 1500, 300), fill=(74, 36, 15))
    g.ellipse((-300, 400, 320, 940), fill=(40, 24, 13))
    img.paste(Image.blend(img, glow.filter(ImageFilter.GaussianBlur(130)), 0.6))

    d = ImageDraw.Draw(img)

    eyebrow, label = fit(d, SEMIBOLD, "VOICE-MATCHED DUBBING", 20, COL_W, spacing="  ")
    d.text((X, 100), label, font=eyebrow, fill=ACCENT)

    y = 186
    for path, line, colour in (
        (SERIF, "Your characters.", CREAM),
        (SERIF, "Any language.", CREAM),
        (SERIF_I, "Their actual voices.", ACCENT),
    ):
        font, _ = fit(d, path, line, 66, COL_W)
        d.text((X, y), line, font=font, fill=colour)
        y += int(font.size * 1.30)

    d.text((X, 452), "23 languages · one command", font=ImageFont.truetype(REGULAR, 24), fill=MUTED)
    d.text((X, 486), "nothing leaves your machine", font=ImageFont.truetype(REGULAR, 24), fill=MUTED)

    d.line((X, 546, X + 52, 546), fill=ACCENT, width=3)
    d.text((X, 568), "github.com/LeftInWalmart/dubtool",
           font=ImageFont.truetype(MONO, 20), fill=MUTED)

    fw, fh = 448, 336
    still = Image.open(ROOT / "demo/media/demo-poster.jpg").convert("RGB").resize((fw, fh), Image.LANCZOS)
    fx, fy = W - fw - 76, 140
    d.rectangle((fx - 1, fy - 1, fx + fw, fy + fh), outline=(255, 255, 255), width=1)
    img.paste(still, (fx, fy))

    d.rectangle((fx, fy + fh - 46, fx + fw, fy + fh), fill=(0, 0, 0))
    small = ImageFont.truetype(SEMIBOLD, 18)
    d.text((fx + 16, fy + fh - 32), "EN", font=small, fill=MUTED)
    d.text((fx + 54, fy + fh - 32), "→", font=small, fill=ACCENT)
    d.text((fx + 76, fy + fh - 32), "ES", font=small, fill=ACCENT)
    d.text((fx + 110, fy + fh - 32), "· dubbed in the actor's own voice", font=small, fill=MUTED)

    # Repo root, not .github/: Explorer hides dot-folders by default, and the upload
    # dialog is a file picker.
    out = ROOT / "social-preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, optimize=True)
    print(f"{out} {out.stat().st_size:,} bytes {img.size}")


if __name__ == "__main__":
    main()
