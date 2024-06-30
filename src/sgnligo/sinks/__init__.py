from sgn.sinks import *
from sgnts.sinks import *
from .kafka_sink import *
from .influx_sink import *
from sgnts.base import Audioadapter
from .. import math


@dataclass
class ImpulseSink(TSSink):
    """
    A fake sink element
    """

    def __post_init__(self):
        super().__post_init__()
        self.cnt = {p: 0 for p in self.sink_pads}
        self.audioadapter = Audioadapter(lib=math)

    def pull(self, pad, bufs):
        """
        getting the buffer on the pad just modifies the name to show this final
        graph point and the prints it to prove it all works.
        """
        self.cnt[pad] += 1
        impulse_position = bufs.metadata["impulse_position"]
        if bufs.EOS:
            self.mark_eos(pad)
        if bufs.buffers is not None:
            print(self.cnt[pad], bufs)
        for buf in bufs:
            if buf.offset > impulse_position - Offset.fromsec(
                1
            ) and buf.offset < impulse_position + Offset.fromsec(
                self.template_duration
            ):
                self.audioadapter.push(buf)
            elif buf.offset > impulse_position + Offset.fromsec(self.template_duration):
                self.impulse_test()
                self.mark_eos(pad)

    @property
    def EOS(self):
        """
        If buffers on any sink pads are End of Stream (EOS), then mark this whole element as EOS
        """
        return any(self.at_eos.values())
