"""Apply the optional Setup Fringes startup position to a Qt window."""

import os
import re


_GEOMETRY = re.compile(r"^(\d+)x(\d+)([+-]\d+)([+-]\d+)$")


def apply_startup_geometry(window):
    value = os.environ.get("ASGARD_SETUP_FRINGES_GEOMETRY", "")
    match = _GEOMETRY.fullmatch(value)
    if match:
        width, height, x, y = map(int, match.groups())
        window.setGeometry(x, y, width, height)
