"""A file that contains some generic plotting module code"""

# Copyright (C) 2013-2016 Kipp Cannon
# Copyright (C) 2015      Chad Hanna
# Copyright (C) 2023      Cort Posnansky
# Copyright (C) 2024      Yun-Jing Huang

from __future__ import annotations

import math
import os
import re
from typing import Optional

import numpy
from matplotlib.colors import hex2color

golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0


def set_matplotlib_cache_directory():
    """Change the matplotlib cache directory to a local temporary location."""

    # FIXME A cleaner solution would be to set MPLCONFIGDIR in the condor
    # submit file, but doing so is either nontrivial or unsupported.
    if os.getenv("_CONDOR_SCRATCH_DIR"):
        os.environ["MPLCONFIGDIR"] = os.getenv("_CONDOR_SCRATCH_DIR")


def colour_from_instruments(
    instruments: list,
    colours: Optional[dict] = None,
):
    """Get the color for a list of instruments.

    Args:
        instruments:
            list, the instruments
        colours:
            dict, the dictionary of colors as numpy arrays with instruments
            as keys

    Returns:
        array, the color
    """

    if colours is None:
        colours = {
            "G1": numpy.array(hex2color("#222222")),
            "H1": numpy.array(hex2color("#ee0000")),
            "L1": numpy.array(hex2color("#4ba6ff")),
            "V1": numpy.array(hex2color("#9b59b6")),
            "K1": numpy.array(hex2color("#ffb200")),
            "E1": numpy.array((1.0, 0.0, 0.0)),
            "E2": numpy.array((0.0, 0.8, 0.0)),
            "E3": numpy.array((1.0, 0.0, 1.0)),
        }
    # mix colours additively
    colour = sum(map(colours.__getitem__, instruments))
    # use single-instrument colours as-given
    if len(instruments) > 1:
        # desaturate
        colour += len(instruments) - 1
        # normalize
        colour /= colour.max()
    return colour


#
# =============================================================================
#
#                                 TeX Helpers
#
# =============================================================================
#


floatpattern = re.compile("([+-]?[.0-9]+)[Ee]([+-]?[0-9]+)")


def latexnumber(s: str):
    """Convert a string of the form "d.dddde-dd" to "d.dddd \\times 10^{-dd}".  Strings
    that contain neither an "e" nor an "E" are returned unchanged.

    Args:
        s:
            str, the sting to convert

    Example:
        >>> import math
        >>> latexnumber("%.12g" % (math.pi * 1e18))
        '3.14159265359 \\\\times 10^{18}'

    Returns:
        str, the converted string
    """

    if "e" not in s and "E" not in s:
        return s
    _match = floatpattern.match(s)
    if _match is None:
        raise ValueError("floatpatterm match failed")
    m, e = _match.groups()
    return r"$%s \times 10^{%d}$" % (m, int(e))


def latexfilename(s: str):
    """Escapes "\\" and "_" characters, and replaces " " with "~" (non-breaking space).

    Args:
        s:
            str, the string to convert

    Returns:
        str, the converted string
    """

    return s.replace("\\", "\\\\").replace("_", "\\_").replace(" ", "~")
