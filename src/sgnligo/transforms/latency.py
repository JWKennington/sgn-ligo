"""An element to calculate latency of buffers."""

# Copyright (C) 2017 Patrick Godwin
# Copyright (C) 2024 Yun-Jing Huang

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sgn.base import TransformElement
from sgnts.base import EventBuffer, EventFrame, TSFrame

from sgnligo.base import now


@dataclass
class Latency(TransformElement):
    """Calculate latency and prepare data into the format expected by the KafkaSink

    Args:
        route:
            str, the kafka route to send the latency data to
        interval:
            float, the interval to calculate latency, in seconds
    """

    route: Optional[str] = None
    interval: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        assert len(self.sink_pads) == 1
        assert isinstance(self.route, str)
        self.frame = None

        if self.interval is not None:
            self.last_time = now()
            self.latencies = []

    def pull(self, pad, frame):
        self.frame = frame

    def new(self, pad):
        """Calculate buffer latency. Latency is defined as the current time subtracted
        by the buffer start time.
        """

        frame = self.frame
        time = now().ns()
        if isinstance(frame, TSFrame):
            framets = frame.buffers[0].t0
            framete = frame.buffers[-1].end
        elif isinstance(frame, EventFrame):
            framets = next(iter(frame.events.values())).ts
            framete = next(iter(frame.events.values())).te

        latency = (time - framets) / 1_000_000_000

        if self.interval is None:
            event_data = {
                self.route: {
                    "time": [
                        framets / 1_000_000_000,
                    ],
                    "data": [
                        latency,
                    ],
                }
            }
        else:
            self.latencies.append(latency)
            if time / 1e9 - self.last_time >= self.interval:
                event_data = {
                    self.route: {
                        "time": [
                            framets / 1_000_000_000,
                        ],
                        "data": [
                            max(self.latencies),
                        ],
                    }
                }
                self.latencies = []
                self.last_time = time / 1e9
            else:
                event_data = None

        return EventFrame(
            events={"kafka": EventBuffer(ts=framets, te=framete, data=event_data)},
            EOS=frame.EOS,
        )
