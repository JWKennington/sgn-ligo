"""An element to track horizon distance for incoming PSDs."""

# Copyright (C) 2024 Becca Ewing, Yun-Jing Huang

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import lal
from sgnts.base import EventBuffer, EventFrame, Offset, TSTransform


@dataclass
class HorizonDistanceTracker(TSTransform):
    """
    Compute horizon distance for an incoming PSD and a given waveform model
    """

    horizon_distance_funcs: Callable | dict[str, Callable] = None
    ifo: str = None
    range: bool = False

    def __post_init__(self):
        super().__post_init__()

    def new(self, pad):
        """
        compute horizon distance
        """
        # incoming frame handling
        frame = self.preparedframes[self.sink_pads[0]]
        EOS = frame.EOS
        shape = frame.shape
        offset = frame.offset
        metadata = frame.metadata

        ts = Offset.tons(offset)
        te = Offset.tons(offset + Offset.fromsamples(shape[-1], frame.sample_rate))

        # get spectrum from metadata
        # FIXME: this is a hack since the PSD is a frequency series.
        psd = metadata["psd"]
        if psd is not None:
            assert isinstance(psd, lal.REAL8FrequencySeries)

            if isinstance(self.horizon_distance_funcs, dict):
                dist = {
                    bankid: func(psd, 8)[0]
                    for bankid, func in self.horizon_distance_funcs.items()
                }
            else:
                dist = self.horizon_distance_funcs(psd, 8)[0]

            data = {}
            if self.range is True:
                data["range_history"] = {
                    "time": [float(ts / 1e9)],
                    "data": [float(dist / 2.25)],
                }
            else:
                data["horizon"] = dist
                data["time"] = ts
                data["ifo"] = self.ifo
                data["navg"] = metadata["navg"]
                data["n_samples"] = metadata["n_samples"]
                data["epoch"] = metadata["epoch"]
        else:
            data = {}
            data["horizon"] = None
            data["time"] = ts
            data["ifo"] = self.ifo
            data["navg"] = None
            data["n_samples"] = None
            data["epoch"] = metadata["epoch"]

        if self.range is True:
            events = {"kafka": EventBuffer(ts, te, data)}
        else:
            events = {"data": EventBuffer(ts, te, data)}

        return EventFrame(
            events=events,
            EOS=EOS,
        )
