from dataclasses import dataclass
import torch

from ..base import SeriesBuffer, TSFrame, TSTransform


@dataclass
class SumIndex(TSTransform):
    """
    Change the data type or the device of the data
    """

    sl: list[slice] = None

    def __post_init__(self):
        super().__post_init__()
        for sl in self.sl:
            assert isinstance(sl, slice)

    def transform(self, pad):
        frame = self.preparedframes[self.sink_pads[0]]

        outbufs = []
        for buf in frame:
            if buf.is_gap:
                out = None
            else:
                data = buf.data
                data_all = []
                for sl in self.sl:
                    if sl.stop - sl.start == 1:
                        data_all.append((data[sl.start, :, :]).unsqueeze(0))
                    else:
                        data_all.append(torch.sum(data[sl, :, :], dim=0).unsqueeze(0))

                out = torch.cat(data_all, dim=0)

            outbuf = SeriesBuffer(
                offset=buf.offset,
                sample_rate=buf.sample_rate,
                data=out,
                shape=(len(self.sl),) + buf.shape[-2:],
            )
        outbufs.append(outbuf)

        return TSFrame(buffers=outbufs, EOS=frame.EOS, metadata=frame.metadata)
