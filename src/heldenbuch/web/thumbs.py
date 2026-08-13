"""Small derivatives of library images, for grids.

The originals are print files -- 2.8 MB on average, up to 4096 px -- and the
shelf, the page grid and every picker were loading them to paint tiles a few
hundred pixels wide. A 16-page book cost tens of megabytes and sixteen
full-size decodes per visit. Grids now request `?thumb=1` and get a cached
JPEG a fraction of that size; only the reader and the zoom view fetch
originals.

Thumbnails live under `<library>/.thumbs/`, mirroring the original's relative
path, and are regenerated whenever the original is newer -- the same
freshness rule `solo.py` uses for its crops. The cache directory starts with
a dot so no library glob ever mistakes a derivative for real content.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

#: Longest edge of a thumbnail. Grid tiles render at up to ~320 CSS pixels;
#: 640 keeps them sharp on a 2x display without approaching original sizes.
MAX_EDGE = 640

#: Only these are worth shrinking; everything else is served as it is.
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def thumbable(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_SUFFIXES


def thumbnail(source: Path, library_root: Path) -> Path:
    """The cached thumbnail for `source`, made if missing or stale.

    Falls back to the original on any failure: a grid that shows the full
    file slowly is better than a grid with a hole in it.
    """
    if not thumbable(source) or not source.is_file():
        return source
    try:
        relative = source.resolve().relative_to(library_root.resolve())
    except ValueError:
        return source

    cached = library_root / ".thumbs" / relative.with_suffix(".jpg")
    try:
        if cached.is_file() and cached.stat().st_mtime >= source.stat().st_mtime:
            return cached
    except OSError:
        return source

    try:
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((MAX_EDGE, MAX_EDGE))
            cached.parent.mkdir(parents=True, exist_ok=True)
            # Write-and-rename, like every other file this project writes:
            # a request interrupted mid-write must not leave a truncated
            # JPEG that later requests would serve as a finished thumbnail.
            tmp = cached.with_name(f"{cached.name}.{os.getpid()}.tmp")
            try:
                image.save(tmp, "JPEG", quality=85)
                os.replace(tmp, cached)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
    except OSError:
        return source
    return cached
