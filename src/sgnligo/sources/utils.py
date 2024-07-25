import lal
from lal import LIGOTimeGPS

import os

import time

def now():
    """
    A convenience function to return the current gps time
    """
    return LIGOTimeGPS(lal.UTCToGPS(time.gmtime()), 0)

def from_T050017(url):
    """
    Parse a URL in the style of T050017-00.
    """
    filename, _ = os.path.splitext(url)
    obs, desc, start, dur = filename.split("-")
    return obs, desc, int(start), int(dur)

def state_vector_on_off_bits(bit):
    """
    Format the given bitmask appropriately as an integer
    """
    if isinstance(bit, str):
        if not bit.startswith("0b"):
            bit = "0b" + bit
        bit = int(bit, 2)
    else:
        bit = int(bit)

    return bit

