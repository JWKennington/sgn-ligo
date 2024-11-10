from dataclasses import dataclass

from sgn.base import SourcePad
from sgnts.base import (
    Array,
    ArrayBackend,
    Audioadapter,
    NumpyBackend,
    Offset,
    SeriesBuffer,
    TSFrame,
    TSTransform,
)
from torch.nn.functional import conv1d as Fconv1d


@dataclass
class LLOIDCorrelate(TSTransform):
    """Correlates input data with filters, with certain time delays, and padding for
    upsampling and downsampling

    Args:
        filters:
            Array, the filter to correlate over
        backend:
            type[ArrayBackend], the wrapper around array operations
        uppad:
            int, the upsampling padding, in offsets
        downpad:
            int, the downsampling padding, in offsets
        delays:
            list[int], a list of time delays for each time slice, in offsets
    """

    filters: Array = None
    backend: type[ArrayBackend] = NumpyBackend
    uppad: int = 0
    downpad: int = 0
    delays: list[int] = None

    def __post_init__(self):
        assert self.filters is not None
        self.shape = self.filters.shape
        self.filters = self.filters.view(-1, 1, self.shape[-1])
        super().__post_init__()
        assert (
            len(self.sink_pads) == 1 and len(self.source_pads) == 1
        ), "only one sink_pad and one source_pad is allowed"

        self.audioadapter = Audioadapter(self.backend)
        self.unique_delays = sorted(set(self.delays))
        self.startup = True

    def corr(self, data: Array) -> Array:
        """Correlate the data with filters.

        Args:
            data:
                Array, the data to correlate with the filters

        Returns:
            Array, the output of the correlation
        """
        # FIXME: try supporing numpy?
        return Fconv1d(data, self.filters, groups=data.shape[-2]).view(
            self.shape[:-1] + (-1,)
        )

    def transform(self, pad: SourcePad) -> TSFrame:
        """Correlates incoming frames with filters.

        Args:
            pad:
                SourcePad, the source pad to produce the transformed frames

        Returns:
            TSFrame, the output TSFrame
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
                outbufs.append(
                    SeriesBuffer(
                        offset=this_segment0 + self.uppad,
                        sample_rate=buf.sample_rate,
                        data=None,
                        shape=self.shape[:-1] + (0,),
                    )
                )
            else:
                this_segment1 = buf.end_offset + self.downpad
                if self.startup:
                    this_segment0 = (
                        this_segment1 - (buf.end_offset - buf.offset) - self.downpad
                    )
                    self.startup = False
                else:
                    this_segment0 = this_segment1 - (buf.end_offset - buf.offset)
                this_segment = (this_segment0, this_segment1)

                A.push(buf)

                outs_map = {}
                # Only do the copy for unique delays
                copied_data = False
                earliest = []

                # copy out the unique segments
                for delay in self.unique_delays:
                    # find the segment to copy out
                    cp_segment1 = this_segment1 + self.uppad - delay
                    cp_segment0 = (
                        cp_segment1
                        - (this_segment1 - this_segment0)
                        - Offset.fromsamples(self.shape[-1] - 1, buf.sample_rate)
                    )
                    earliest.append(cp_segment0)
                    if cp_segment1 > A.offset and not A.is_gap():
                        if A.data_all is None:
                            A.concatenate_data(
                                (
                                    max(
                                        this_segment0
                                        + self.uppad
                                        - max(self.unique_delays)
                                        - Offset.fromsamples(
                                            self.shape[-1] - 1, buf.sample_rate
                                        ),
                                        A.offset,
                                    ),
                                    this_segment1
                                    + self.uppad
                                    - min(self.unique_delays),
                                )
                            )
                        cp_segment = (max(A.offset, cp_segment0), cp_segment1)
                        # We need to do a copy
                        out = A.copy_samples_by_offset_segment(cp_segment)
                        if cp_segment0 < A.offset and out is not None:
                            # pad with zeros in front
                            pad_length = Offset.tosamples(
                                A.offset - cp_segment0, buf.sample_rate
                            )
                            out = self.backend.pad(out, (pad_length, 0))
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
                                out = self.backend.zeros(
                                    (
                                        Offset.tosamples(
                                            cp_segment1 - cp_segment0, buf.sample_rate
                                        ),
                                    )
                                )
                            outs.append(out)
                        outs = self.backend.stack(outs)
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
                        shape=self.shape[:-1]
                        + (
                            Offset.tosamples(
                                this_segment1 - this_segment0, buf.sample_rate
                            ),
                        ),
                    )
                )
        return TSFrame(buffers=outbufs, EOS=frame.EOS, metadata=frame.metadata)
