from . import *
from ..base import *
from typing import Any
from torch.nn.functional import conv1d as Fconv1d

import torch
from sgn.base import TransformElement


@dataclass
class TorchCorrelateValid(TransformElement):
    """
    Correlates input data with filters

    Parameters:
    -----------
    filters: Sequence[Any]
        the filter to correlate over

    Assumptions:
    ------------
    - There is only one sink pad and one source pad
    """

    filters: Sequence[Any] = None

    def __post_init__(self):
        assert self.filters is not None
        self.shape = self.filters.shape
        self.filters = self.filters.view(-1, 1, self.shape[-1])
        super().__post_init__()
        assert (
            len(self.sink_pads) == 1 and len(self.source_pads) == 1
        ), "only one sink_pad and one source_pad is allowed"

    def pull(self, pad, frame):
        self.frame = frame
        if frame.shape[-1] > 0:
            assert frame.shape[-1] > self.shape[-1]

    def corr(self, data):
        return Fconv1d(data, self.filters, groups=data.shape[-2]).view(
            self.shape[:-1] + (-1,)
        )

    def transform(self, pad):
        """
        Correlates data with filters
        """
        outbufs = []
        frame = self.frame
        offset=frame.offset + Offset.fromsamples(self.shape[-1] - 1, frame.sample_rate)
        if frame.shape[-1] == 0:
            #offset=frame.offset
            #+ Offset.fromsamples(self.shape[-1] - 1, frame.sample_rate),
            outbufs.append(
                SeriesBuffer(
                    offset=frame.offset + Offset.fromsamples(self.shape[-1] - 1, frame.sample_rate),
                    sample_rate=frame.sample_rate,
                    data=None,
                    shape=self.shape[:-1] + (0,),
                )
            )
            return TSFrame(buffers=outbufs, EOS=frame.EOS, metadata=frame.metadata)

        for i, buf in enumerate(frame):
            if buf.is_gap:
                data = None
            else:
                # FIXME: Are there multi-channel correlation in numpy or scipy?
                # FIXME: consider multi-dimensional filters
                data = self.corr(buf.data)
            shape=self.shape[:-1] + (buf.samples - self.shape[-1] + 1,)
            outbufs.append(
                SeriesBuffer(
                    offset=buf.offset
                    + Offset.fromsamples(self.shape[-1] - 1, buf.sample_rate),
                    sample_rate=buf.sample_rate,
                    data=data,
                    shape=self.shape[:-1] + (buf.samples - self.shape[-1] + 1,),
                )
            )
        return TSFrame(buffers=outbufs, EOS=frame.EOS, metadata=frame.metadata)
