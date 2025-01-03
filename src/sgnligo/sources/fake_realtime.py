from dataclasses import dataclass

from sgnts.sources import RealTimeWhiteNoiseSource

from sgnligo.base import now as gps_now


@dataclass
class RealTimeWhiteNoiseGPSSource(RealTimeWhiteNoiseSource):
    """A time-series source that generates fake data in fixed-size buffers in real-time
    and in GPS time
    """

    def time_now(self):
        return float(gps_now())
