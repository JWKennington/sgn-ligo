from dataclasses import dataclass
from sgnts.base import TSTransform, SeriesBuffer, TSFrame
from ..base import now


@dataclass
class Latency(TSTransform):
    """
    Calculate latency
    """

    route: str = None

    def __post_init__(self):
        super().__post_init__()
        assert len(self.sink_pads) == 1
        assert isinstance(self.route, str)

    def transform(self, pad):
        frame = self.preparedframes[self.sink_pads[0]]
        metadata = frame.metadata
        time = now().ns()
        latency = (time - frame.buffers[0].t0) / 1_000_000_000

        outbuf = SeriesBuffer(
            offset=frame.offset,
            sample_rate=frame.sample_rate,
            data=None,
            shape=frame.shape,
        )
        if "kafka" not in metadata:
            metadata["kafka"] = {}
        metadata["kafka"][self.route] = {
            "time": [
                frame.buffers[0].t0 / 1_000_000_000,
            ],
            "data": [
                latency,
            ],
        }

        return TSFrame(
            buffers=[outbuf],
            metadata=metadata,
            EOS=frame.EOS,
        )
