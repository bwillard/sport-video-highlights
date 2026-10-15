"""
Title cards: white text on a dark background, drawn with Pillow and turned
into a short silent clip by the renderer.

- Opening card: the reel title, then for a one-game reel the game's
  description and date, or for a multi-game reel the number of games and
  the date range.
- Clip card (before each clip): the time in the game, and the clip's note if
  it has one, so viewers know what they're about to see.
"""

from datetime import date
from functools import cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from soccer_clips.models import Clip, Game, Reel

# How long a card stays up: time to take in the card, plus time to read its
# text at subtitle speed. Subtitle guides allow 17-20 characters/second for
# adults and less for children (Netflix's English guide: 20 adult, 17
# children); 15 leaves room for young viewers and for a glance at the game
# time. Every card is shown for at least the reel's minimum (Reel.card_s).
READING_CPS = 15
ORIENT_S = 1.0
MAX_CARD_S = 15.0

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


def card_seconds(lines: list[str], min_s: float) -> float:
    """Seconds to show a card with these lines: long enough to read, at least min_s."""
    chars = sum(len(line.strip()) for line in lines)
    return min(MAX_CARD_S, max(min_s, ORIENT_S + chars / READING_CPS))


def long_date(d: date) -> str:
    """Saturday, October 10, 2026"""
    return f"{d:%A, %B} {d.day}, {d.year}"


def date_range(first: date, last: date) -> str:
    if first == last:
        return long_date(first)
    if first.year == last.year:
        return f"{first:%B} {first.day} – {last:%B} {last.day}, {last.year}"
    return f"{first:%B} {first.day}, {first.year} – {last:%B} {last.day}, {last.year}"


def start_card_lines(reel: Reel, games: list[Game]) -> list[str]:
    """Title, then game description + date (one game) or "N games · date range" (several)."""
    lines = [reel.title]
    unique = list({g.id: g for g in games}.values())
    if len(unique) == 1:
        g = unique[0]
        description = (g.description or "").strip()
        if description and description.casefold() != reel.title.strip().casefold():
            lines.append(description)
        if g.played_on:
            lines.append(long_date(g.played_on))
    elif unique:
        dates = sorted(g.played_on for g in unique if g.played_on)
        summary = f"{len(unique)} games"
        lines.append(f"{summary} · {date_range(dates[0], dates[-1])}" if dates else summary)
    return lines


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
    First line is the headline (title or game time); each further line is
    wrapped body text (game description, date, or the clip's note). Everything
    is centred; sizes scale with the height.
    """
    img = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(img)
    max_w = width * 0.8
    head_font = _font(round(height * 0.11))
    body_font = _font(round(height * 0.055))
    gap = round(height * 0.04)

    head = _wrap(lines[0], head_font, max_w, 2) if lines else []
    body = [part for line in lines[1:] for part in _wrap(line, body_font, max_w, MAX_NOTE_LINES)]
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
