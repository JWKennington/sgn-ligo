from dataclasses import dataclass

from ..base import SeriesBuffer, TSFrame, TSTransform, Audioadapter, Offset


@dataclass
class TimeShifter(TSTransform):
    """
    Change the data type or the device of the data
    """

    offset_segments: list[tuple[float, float]] = None
    shift: int = None
    lib: int = None

    def __post_init__(self):
        super().__post_init__()
        self.audioadapter = Audioadapter(self.lib)
        self.unique_segments = self.offset_segments
        self.noffset = set(seg[1] - seg[0] for seg in self.unique_segments)
        assert len(self.noffset) == 1, "segments must be same length"
        self.noffset = list(self.noffset)[0]
        self.earliest_offset = min(o[0] for o in self.unique_segments)

    def pull(self, pad, frame):
        super().pull(pad, frame)
        for buf in frame:
            self.audioadapter.push(buf)

    def transform(self, pad):
        A = self.audioadapter
        frame = self.preparedframes[self.sink_pads[0]]
        # use the offset segment from the new frame as reference
        newest_offset = frame.end_offset
        A.concatenate_data()

        outs_map = {}
        # Only do the copy for unique segments
        copied_data = False
        for segment in self.unique_segments:
            cp_segment1 = newest_offset + segment[1]
            cp_segment0 = newest_offset + segment[0]
            if cp_segment1 > A.offset:
                cp_segment = (max(A.offset, cp_segment0), cp_segment1)
                # We need to do a copy
                out = A.copy_samples_by_offset_segment(cp_segment)
                if cp_segment0 < A.offset and out is not None:
                    # pad with zeros in front
                    pad_length = Offset.tosamples(
                        A.offset - cp_segment0, frame.sample_rate
                    )
                    out = self.lib.pad_func(out, (pad_length, 0))
                copied_data = True
            else:
                out = None
            outs_map[segment] = out

        if copied_data is True:
            outs = []
            # Now stack the output array
            for segment in self.offset_segments:
                out = outs_map[segment]
                if out is None:
                    out = self.lib.zeros_func(
                        (Offset.tosamples(segment[1] - segment[0], frame.sample_rate),)
                    )
                outs.append(out)

            outs = self.lib.stack_func(outs)
        else:
            outs = None

        flush_end_offset = newest_offset + self.earliest_offset
        if flush_end_offset > A.offset:
            A.flush_samples_by_end_offset_segment(flush_end_offset)

        outbuf = SeriesBuffer(
            offset=newest_offset + self.shift - self.noffset,
            sample_rate=frame.sample_rate,
            data=outs,
            shape=(
                len(self.offset_segments),
                Offset.tosamples(cp_segment1 - cp_segment0, frame.sample_rate),
            ),
        )

        return TSFrame(
            buffers=[outbuf],
            EOS=frame.EOS,
        )
