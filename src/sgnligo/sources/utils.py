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

