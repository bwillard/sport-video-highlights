"""
Title cards: white text on a dark background, drawn with Pillow and turned
into a short silent clip by the renderer.

- Opening card: the reel title.
- Clip card (before each clip): the time in the game, and the clip's note if
  it has one, so viewers know what they're about to see.
"""

from functools import cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from soccer_clips.models import Clip, Game, Reel

BACKGROUND = (17, 17, 17)
TEXT = (255, 255, 255)
SUBTEXT = (210, 210, 210)
MAX_NOTE_LINES = 4

# A bold system font if there is one; otherwise Pillow's built-in font.
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


@cache
def font_path() -> str | None:
    return next((p for p in FONT_CANDIDATES if Path(p).exists()), None)


@cache
def _font(size: int) -> ImageFont.FreeTypeFont:
    path = font_path()
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size)


def start_card_lines(reel: Reel) -> list[str]:
    return [reel.title]


def clip_card_lines(game: Game, clip: Clip) -> list[str]:
    """[time in the game, note (if any)]."""
    lines = [game.time_label(clip.start_s)]
    if clip.note and clip.note.strip():
        lines.append(clip.note.strip())
    return lines


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: float, max_lines: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if font.getlength(candidate) <= max_width or not line:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and font.getlength(last + "…") > max_width:
            last = last[:-1].rstrip()
        lines[-1] = last + "…"
    return lines


def draw_card(lines: list[str], width: int, height: int) -> Image.Image:
    """
    First line is the headline (title or game time), the rest is wrapped body
    text (the note). Everything is centred; sizes scale with the height.
    """
    img = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(img)
    max_w = width * 0.8
    head_font = _font(round(height * 0.11))
    body_font = _font(round(height * 0.055))
    gap = round(height * 0.04)

    head = _wrap(lines[0], head_font, max_w, 2) if lines else []
    body = _wrap(" ".join(lines[1:]), body_font, max_w, MAX_NOTE_LINES) if len(lines) > 1 else []
    blocks = [(t, head_font, TEXT) for t in head] + [(t, body_font, SUBTEXT) for t in body]

    def line_h(font: ImageFont.FreeTypeFont) -> int:
        top, bottom = font.getbbox("Ag")[1::2]
        return round((bottom - top) * 1.3)

    total = sum(line_h(f) for _, f, _ in blocks) + (gap if head and body else 0)
    y = (height - total) / 2
    for i, (text, font, colour) in enumerate(blocks):
        if body and i == len(head):
            y += gap
        draw.text((width / 2, y), text, font=font, fill=colour, anchor="mt")
        y += line_h(font)
    return img
