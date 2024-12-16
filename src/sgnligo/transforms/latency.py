"""An element to calculate latency of buffers."""

# Copyright (C) 2017 Patrick Godwin
# Copyright (C) 2024 Yun-Jing Huang

from dataclasses import dataclass

from sgn.base import TransformElement
from sgnts.base import EventBuffer, EventFrame, TSFrame

from sgnligo.base import now


@dataclass
class Latency(TransformElement):
    """Calculate latency and prepare data into the format expected by the KafkaSink

    Args:
        route:
            str, the kafka route to send the latency data to
    """

    route: str = None

    def __post_init__(self):
        super().__post_init__()
        assert len(self.sink_pads) == 1
        assert isinstance(self.route, str)
        self.frame = None

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

        return EventFrame(
            events={"kafka": EventBuffer(ts=framets, te=framete, data=event_data)},
            EOS=frame.EOS,
        )
