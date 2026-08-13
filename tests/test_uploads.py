"""Tests for handling what comes out of a browser file picker.

A phone photo carries its rotation in an EXIF tag rather than in the pixels. If
that is not applied, a portrait photo arrives on its side and the character
gets drawn sideways with nothing in the log to explain why -- the kind of bug
that is invisible until you look at the output and then obvious forever.
"""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from storytime.web.bookjobs import MAX_UPLOAD_BYTES, save_upload

ORIENTATION_TAG = 274


def _upload(image: Image.Image, name: str = "photo.jpg", orientation: int | None = None) -> dict:
    buffer = io.BytesIO()
    if orientation is not None:
        exif = image.getexif()
        exif[ORIENTATION_TAG] = orientation
        image.save(buffer, format="JPEG", exif=exif)
    else:
        image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"name": name, "data": f"data:image/jpeg;base64,{encoded}"}


def _landscape() -> Image.Image:
    """A wide image with a marker in the top-left, so rotation is detectable."""
    image = Image.new("RGB", (120, 60), (30, 90, 180))
    image.paste(Image.new("RGB", (20, 20), (240, 40, 40)), (0, 0))
    return image


def test_upload_is_written_and_readable(tmp_path):
    target = save_upload(_upload(_landscape()), tmp_path, "photo_1")
    assert target is not None and target.is_file()
    with Image.open(target) as image:
        assert image.size == (120, 60)


@pytest.mark.parametrize("orientation,expected", [(6, (60, 120)), (8, (60, 120)), (3, (120, 60))])
def test_exif_rotation_is_applied_to_the_pixels(tmp_path, orientation, expected):
    """Orientation 6 and 8 are quarter turns, so width and height swap."""
    target = save_upload(_upload(_landscape(), orientation=orientation), tmp_path, "photo_1")
    with Image.open(target) as image:
        assert image.size == expected


def test_rotation_actually_moves_the_content(tmp_path):
    """Not just the dimensions -- the marker has to end up somewhere new."""
    target = save_upload(_upload(_landscape(), orientation=6), tmp_path, "photo_1")
    with Image.open(target) as image:
        upright = image.convert("RGB")
    # A 90 degree clockwise turn sends the top-left marker to the top-right.
    top_right = upright.getpixel((upright.width - 6, 6))
    assert top_right[0] > 180 and top_right[1] < 100, "expected the red marker top-right"


def test_saved_file_keeps_no_orientation_tag(tmp_path):
    """Otherwise a downstream viewer would rotate it a second time."""
    target = save_upload(_upload(_landscape(), orientation=6), tmp_path, "photo_1")
    with Image.open(target) as image:
        assert image.getexif().get(ORIENTATION_TAG) in (None, 1)


def test_unknown_suffix_falls_back_to_jpg(tmp_path):
    target = save_upload(_upload(_landscape(), name="scan.heic"), tmp_path, "photo_1")
    assert target is not None and target.suffix == ".jpg"


def test_png_keeps_its_suffix(tmp_path):
    target = save_upload(_upload(_landscape(), name="drawing.png"), tmp_path, "ref")
    assert target is not None and target.suffix == ".png"


def test_garbage_payload_is_rejected_rather_than_written(tmp_path):
    assert save_upload({"name": "x.jpg", "data": "not base64 at all!!"}, tmp_path, "p") is None
    assert save_upload({"name": "x.jpg", "data": ""}, tmp_path, "p") is None
    assert save_upload({}, tmp_path, "p") is None


def test_oversized_upload_is_refused(tmp_path):
    payload = base64.b64encode(b"\0" * (MAX_UPLOAD_BYTES + 1024)).decode("ascii")
    assert save_upload({"name": "big.jpg", "data": payload}, tmp_path, "p") is None


def test_a_file_that_is_not_an_image_is_left_alone_not_crashed(tmp_path):
    payload = base64.b64encode(b"this is plain text, not a picture").decode("ascii")
    target = save_upload({"name": "notes.jpg", "data": payload}, tmp_path, "p")
    assert target is not None and target.is_file()
