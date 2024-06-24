import lal
from lal import LIGOTimeGPS
import time

def now():
	"""
	A convenience function to return the current gps time
	"""
	return LIGOTimeGPS(lal.UTCToGPS(time.gmtime()), 0)


