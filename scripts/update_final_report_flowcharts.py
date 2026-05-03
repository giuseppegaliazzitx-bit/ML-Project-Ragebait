"""Replace final report embedded flowchart images with regenerated assets.

This is a fallback for environments without a LaTeX engine. The TeX report
already references these PNG filenames, so replacing the matching PDF image
objects keeps the report synchronized with the top-level final assets.
"""

from __future__ import annotations

import os
from pathlib import Path

import fitz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PDF = PROJECT_ROOT / "deliverables" / "final_report.pdf"
ASSET_DIR = PROJECT_ROOT / "deliverables" / "final_assets"


TARGETS = [
    {
        "page_index": 4,
        "size": (2577, 1011),
        "asset": ASSET_DIR / "pipeline_flowchart.png",
        "name": "pipeline flowchart",
    },
    {
        "page_index": 9,
        "size": (2585, 1011),
        "asset": ASSET_DIR / "task_framing_flowchart.png",
        "name": "task framing flowchart",
    },
]


def find_image_xref(doc: fitz.Document, page_index: int, size: tuple[int, int]) -> int:
    page = doc[page_index]
    matches: set[int] = set()
    for image in page.get_images(full=True):
        xref = image[0]
        info = doc.extract_image(xref)
        if (info["width"], info["height"]) == size:
            matches.add(xref)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {size[0]}x{size[1]} image on page {page_index + 1}; found {len(matches)}."
        )
    return next(iter(matches))


def main() -> None:
    doc = fitz.open(REPORT_PDF)
    for target in TARGETS:
        xref = find_image_xref(doc, target["page_index"], target["size"])
        doc[target["page_index"]].replace_image(xref, filename=target["asset"])
        print(f"Replaced {target['name']} image xref {xref} on page {target['page_index'] + 1}.")

    tmp_path = REPORT_PDF.with_suffix(".updated.pdf")
    doc.save(tmp_path, garbage=4, deflate=True)
    doc.close()
    os.replace(tmp_path, REPORT_PDF)


if __name__ == "__main__":
    main()
