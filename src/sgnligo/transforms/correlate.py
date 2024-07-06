from . import *
from ..base import *
from typing import Any
from torch.nn.functional import conv1d as Fconv1d

import torch
from sgnts.base import TSTransform


@dataclass
class LLOIDCorrelate(TSTransform):
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
    lib: int = None
    uppad: int = None
    downpad: int = None
    delays: int = None

    def __post_init__(self):
        assert self.filters is not None
        self.shape = self.filters.shape
        self.filters = self.filters.view(-1, 1, self.shape[-1])
        super().__post_init__()
        assert (
            len(self.sink_pads) == 1 and len(self.source_pads) == 1
        ), "only one sink_pad and one source_pad is allowed"

        self.audioadapter = Audioadapter(self.lib)
        self.unique_delays = sorted(set(self.delays))
        self.startup = True

    def corr(self, data):
        return Fconv1d(data, self.filters, groups=data.shape[-2]).view(
            self.shape[:-1] + (-1,)
        )

    def transform(self, pad):
        """
        Correlates data with filters
        """
        A = self.audioadapter
        frame = self.preparedframes[self.sink_pads[0]]
        outbufs = []

        # process buffer by buffer
        for buf in frame:
            # find the reference segment "this_segment"
            if buf.end_offset - buf.offset == 0:
                if self.startup:
                    this_segment1 = buf.end_offset
                else:
                    this_segment1 = buf.end_offset + self.downpad
                this_segment0 = this_segment1
                outbufs.append(SeriesBuffer(
                        offset=this_segment0 + self.uppad,
                        sample_rate=buf.sample_rate,
                        data=None,
                        shape=self.shape[:-1] + (0,),
                    )
                )
            else:
                this_segment1 = buf.end_offset + self.downpad
                if self.startup:
                    this_segment0 = this_segment1 - (buf.end_offset - buf.offset) - self.downpad
                    self.startup = False
                else:
                    this_segment0 = this_segment1 - (buf.end_offset - buf.offset)
                this_segment = (this_segment0, this_segment1)
                nthis_segment = this_segment1 - this_segment0

                A.push(buf)

                outs_map = {}
                # Only do the copy for unique delays 
                copied_data = False
                earliest = []

                # copy out the unique segments
                for delay in self.unique_delays:
                    # find the segment to copy out
                    cp_segment1 = this_segment1 + self.uppad - delay
                    cp_segment0 = cp_segment1 - (this_segment1 - this_segment0) - Offset.fromsamples(self.shape[-1] - 1, buf.sample_rate)
                    earliest.append(cp_segment0)
                    if cp_segment1 > A.offset and not A.is_gap():
                        if A.data_all is None:
                            A.concatenate_data()
                        cp_segment = (max(A.offset, cp_segment0), cp_segment1)
                        # We need to do a copy
                        out = A.copy_samples_by_offset_segment(cp_segment)
                        if cp_segment0 < A.offset and out is not None:
                            # pad with zeros in front
                            pad_length = Offset.tosamples(
                                A.offset - cp_segment0, buf.sample_rate
                            )
                            out = self.lib.pad_func(out, (pad_length, 0))
                        copied_data = True
                    else:
                        out = None
                    outs_map[delay] = out

                # fill in zeros arrays
                if copied_data is True:
                    outs = []
                    # Now stack the output array
                    if len(self.unique_delays) == 1:
                        outs = outs_map[delay].unsqueeze(0)
                    else:
                        for delay in self.delays:
                            out = outs_map[delay]
                            if out is None:
                               out = self.lib.zeros_func(
                                    (Offset.tosamples(cp_segment1 - cp_segment0, buf.sample_rate),)
                                )
                            outs.append(out)
                        outs = self.lib.stack_func(outs)
                else:
                    outs = None

                # flush data
                flush_end_offset = min(earliest)
                if flush_end_offset > A.offset:
                    A.flush_samples_by_end_offset_segment(flush_end_offset)
                A.data_all = None

                # Do the correlation!
                if outs is not None:
                    outs = self.corr(outs)
                outbufs.append(
                    SeriesBuffer(
                        offset=this_segment[0] + self.uppad,
                        sample_rate=buf.sample_rate,
                        data=outs,
                        shape=self.shape[:-1] + (Offset.tosamples(this_segment1 - this_segment0, buf.sample_rate),),
                    )
                )
        return TSFrame(buffers=outbufs, EOS=frame.EOS, metadata=frame.metadata)
