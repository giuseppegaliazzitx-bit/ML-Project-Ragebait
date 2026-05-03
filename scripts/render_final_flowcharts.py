"""Render final report flowchart PNGs.

The original flowchart PNGs were raster-only deliverables. This script keeps a
small reproducible source for the two report graphics and writes the same asset
filenames used by the LaTeX report and slide HTML.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "deliverables" / "final_assets"

BG = "#f7f9fc"
INK = "#172033"
MUTED = "#4f5f73"
LINE = "#64748b"
WHITE = "#ffffff"
BLUE = "#2563eb"
BLUE_SOFT = "#dbeafe"
TEAL = "#0f766e"
TEAL_SOFT = "#ccfbf1"
AMBER = "#b45309"
AMBER_SOFT = "#fef3c7"
SLATE_SOFT = "#e2e8f0"
GREEN = "#15803d"
GREEN_SOFT = "#dcfce7"
RED = "#b91c1c"
RED_SOFT = "#fee2e2"
PURPLE = "#7c3aed"
PURPLE_SOFT = "#ede9fe"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def font_from(base: ImageFont.FreeTypeFont, size: int) -> ImageFont.FreeTypeFont:
    path = getattr(base, "path", None)
    if path:
        return ImageFont.truetype(path, size)
    return font(size)


TITLE = font(54, True)
SUBTITLE = font(30)
LABEL = font(33, True)
BODY = font(25)
SMALL = font(22)


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int
    title: str
    body: str
    stroke: str
    fill: str


def wrap_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    typeface: ImageFont.FreeTypeFont,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        probe = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), probe, font=typeface)[2] <= max_width:
            current = probe
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def line_height(draw: ImageDraw.ImageDraw, typeface: ImageFont.FreeTypeFont, spacing: int) -> int:
    bbox = draw.textbbox((0, 0), "Ag", font=typeface)
    return bbox[3] - bbox[1] + spacing


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    typeface: ImageFont.FreeTypeFont,
    fill: str,
    *,
    spacing: int = 8,
    align: str = "center",
    min_size: int = 17,
) -> None:
    x0, y0, x1, y1 = box
    max_width = x1 - x0
    max_height = y1 - y0
    size = getattr(typeface, "size", 24)
    face = typeface
    while True:
        lines = wrap_lines(draw, text, max_width, face)
        lh = line_height(draw, face, spacing)
        total_height = len(lines) * lh - spacing
        widest = max((draw.textbbox((0, 0), line, font=face)[2] for line in lines), default=0)
        if (total_height <= max_height and widest <= max_width) or size <= min_size:
            break
        size -= 1
        face = font_from(typeface, size)

    y = y0 + max((max_height - total_height) // 2, 0)
    for line in lines:
        width = draw.textbbox((0, 0), line, font=face)[2]
        if align == "left":
            x = x0
        else:
            x = x0 + (max_width - width) // 2
        draw.text((x, y), line, font=face, fill=fill)
        y += lh


def draw_box(draw: ImageDraw.ImageDraw, box: Box) -> None:
    shadow = (box.x + 8, box.y + 10, box.x + box.w + 8, box.y + box.h + 10)
    rect = (box.x, box.y, box.x + box.w, box.y + box.h)
    draw.rounded_rectangle(shadow, radius=22, fill="#d9e2ef")
    draw.rounded_rectangle(rect, radius=22, fill=box.fill, outline=box.stroke, width=4)
    draw_wrapped(draw, box.title, (box.x + 36, box.y + 24, box.x + box.w - 36, box.y + 74), LABEL, INK, min_size=24)
    draw.line((box.x + 40, box.y + 92, box.x + box.w - 40, box.y + 92), fill=box.stroke, width=3)
    draw_wrapped(draw, box.body, (box.x + 42, box.y + 110, box.x + box.w - 42, box.y + box.h - 30), BODY, MUTED, spacing=7)


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = LINE) -> None:
    draw.line((*start, *end), fill=color, width=7)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex >= sx else -1
        points = [(ex, ey), (ex - direction * 24, ey - 15), (ex - direction * 24, ey + 15)]
    else:
        direction = 1 if ey >= sy else -1
        points = [(ex, ey), (ex - 15, ey - direction * 24), (ex + 15, ey - direction * 24)]
    draw.polygon(points, fill=color)


def draw_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str, width: int) -> None:
    draw.text((90, 58), title, font=TITLE, fill=INK)
    draw.text((92, 126), subtitle, font=SUBTITLE, fill=MUTED)
    draw.line((90, 184, width - 90, 184), fill="#cbd5e1", width=3)


def render_pipeline() -> None:
    width, height = 2577, 1011
    image = Image.new("RGBA", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw_header(
        draw,
        "Final Iteration 2 Modeling Pipeline",
        "Human labels are frozen once, then reused across comparable model families and final test claims.",
        width,
    )

    boxes = [
        Box(80, 265, 390, 210, "Gold-label data", "12,490 human-labeled short posts with Normal, Profanity, Trolling, Derogatory, and Hate Speech labels.", TEAL, TEAL_SOFT),
        Box(560, 265, 390, 210, "Frozen splits", "Stratified 80/10/10 train, validation, and held-out test splits using seed 42.", BLUE, BLUE_SOFT),
        Box(1040, 215, 430, 165, "TF-IDF features", "20k unigram and bigram lexical features.", AMBER, AMBER_SOFT),
        Box(1040, 430, 430, 165, "Train-only tokens", "Mean-pooled embeddings for the FFNN baseline.", GREEN, GREEN_SOFT),
        Box(1040, 645, 430, 165, "BERT tokens", "WordPiece inputs for fine-tuned BERT classifiers.", PURPLE, PURPLE_SOFT),
        Box(1585, 265, 390, 210, "Model ladder", "Linear baselines, shallow neural baseline, and contextual BERT classifiers for binary and multiclass tasks.", BLUE, WHITE),
        Box(2065, 265, 390, 210, "Evaluation package", "Validation selection, held-out test metrics, confusion matrices, timing, predictions, and error analysis artifacts.", RED, RED_SOFT),
    ]
    for b in boxes:
        draw_box(draw, b)

    draw_arrow(draw, (470, 370), (560, 370))
    draw_arrow(draw, (950, 370), (1040, 297))
    draw_arrow(draw, (950, 370), (1040, 512))
    draw_arrow(draw, (950, 370), (1040, 727))
    draw_arrow(draw, (1470, 297), (1585, 370))
    draw_arrow(draw, (1470, 512), (1585, 370))
    draw_arrow(draw, (1470, 727), (1585, 370))
    draw_arrow(draw, (1975, 370), (2065, 370))

    draw.rounded_rectangle((1040, 845, 1975, 920), radius=18, fill=WHITE, outline="#cbd5e1", width=3)
    draw_wrapped(
        draw,
        "Same splits and metrics make model comparisons auditable instead of slide-specific.",
        (1080, 857, 1935, 908),
        SMALL,
        MUTED,
    )

    image.save(ASSET_DIR / "pipeline_flowchart.png")


def render_task_framing() -> None:
    width, height = 2585, 1011
    image = Image.new("RGBA", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw_header(
        draw,
        "Two Task Framings From One Gold-label Corpus",
        "The same human labels support a high-recall moderation gate and a harder diagnostic classifier.",
        width,
    )

    source = Box(90, 380, 430, 230, "Human labels", "Normal, Profanity, Trolling, Derogatory, and Hate Speech are preserved as the source of truth.", TEAL, TEAL_SOFT)
    binary = Box(770, 245, 510, 225, "Binary detector", "Normal maps to 0. All non-Normal labels map to Ragebait / Abusive = 1.", BLUE, BLUE_SOFT)
    multi = Box(770, 585, 510, 225, "Five-class classifier", "Each original label remains separate for behavior-specific diagnosis.", PURPLE, PURPLE_SOFT)
    out1 = Box(1545, 245, 430, 225, "Coarse decision", "Flag versus do not flag. Best for screening and triage workflows.", AMBER, AMBER_SOFT)
    out2 = Box(1545, 585, 430, 225, "Diagnostic label", "Normal, Profanity, Trolling, Derogatory, or Hate Speech.", GREEN, GREEN_SOFT)
    final = Box(2115, 415, 370, 230, "Reported claims", "Binary F1 = 0.9222. Multiclass macro F1 = 0.6405.", RED, RED_SOFT)
    for b in [source, binary, multi, out1, out2, final]:
        draw_box(draw, b)

    draw_arrow(draw, (520, 495), (770, 355))
    draw_arrow(draw, (520, 495), (770, 695))
    draw_arrow(draw, (1280, 355), (1545, 355))
    draw_arrow(draw, (1280, 695), (1545, 695))
    draw_arrow(draw, (1975, 355), (2115, 500))
    draw_arrow(draw, (1975, 695), (2115, 560))

    draw.rounded_rectangle((760, 860, 1985, 930), radius=18, fill=WHITE, outline="#cbd5e1", width=3)
    draw_wrapped(
        draw,
        "Binary is easier and operationally useful; multiclass is harder but reveals where semantic boundaries still fail.",
        (810, 873, 1935, 917),
        SMALL,
        MUTED,
    )

    image.save(ASSET_DIR / "task_framing_flowchart.png")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    render_pipeline()
    render_task_framing()


if __name__ == "__main__":
    main()
