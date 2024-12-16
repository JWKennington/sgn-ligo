"""An element to apply a bit mask on the incoming data."""

# Copyright (C) 2011-2012,2014,2015 Kipp Cannon
# Copyright (C) 2024 Yun-Jing Huang

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sgn.base import SourcePad
from sgnts.base import SeriesBuffer, TSFrame, TSTransform

from sgnligo.base import state_vector_on_off_bits


@dataclass
class BitMask(TSTransform):
    """Apply the bit mask on the incoming data.

    Args:
        bit_mask:
            int, the bit mask to apply on the data. Data is passed through if
            data_bit & bit_mask == bit_mask. The data not matching the bit mask will
            be turned into gap buffers.
    """

    bit_mask: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()

        assert self.bit_mask is not None

        # set state vector on bits
        self.bit_mask = state_vector_on_off_bits(self.bit_mask)

    def new(self, pad: SourcePad) -> TSFrame:
        """Produce non-gap buffers if the bit of the data passes the bit mask,
        otherwise, produce gap buffers.

        Args:
            pad:
                SourcePad, the source pad to produce TSFrames

        Returns:
            TSFrame, the TSFrame carrying buffers with bit mask applied
        """
        frame = self.preparedframes[self.sink_pads[0]]
        metadata = frame.metadata

        outbufs = []
        for buf in frame:
            if not buf.is_gap:
                buf_offset = buf.offset
                sample_rate = buf.sample_rate

                state_flags = [
                    state_vector_on_off_bits(b) & self.bit_mask == self.bit_mask
                    for b in buf.data
                ]

                if all(state_flags):
                    # Pass through
                    outbufs.append(buf)
                elif not any(state_flags):
                    # Full gap
                    buf.set_data(None)
                    outbufs.append(buf)
                else:
                    for b in buf.data:
                        bit = state_vector_on_off_bits(b)
                        if bit & self.bit_mask == self.bit_mask:
                            outbuf = SeriesBuffer(
                                offset=buf_offset,
                                sample_rate=sample_rate,
                                data=np.array([b]),
                                shape=(1,),
                            )
                        else:
                            outbuf = SeriesBuffer(
                                offset=buf_offset,
                                sample_rate=sample_rate,
                                data=None,
                                shape=(1,),
                            )
                        outbufs.append(outbuf)
                        buf_offset = outbuf.end_offset
            else:
                outbufs.append(buf)

        return TSFrame(
            buffers=outbufs,
            metadata=metadata,
            EOS=frame.EOS,
        )
