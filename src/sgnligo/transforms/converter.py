from dataclasses import dataclass

from ..base import SeriesBuffer, TSFrame, TSTransform

import torch
import numpy as np


@dataclass
class Converter(TSTransform):
    """
    Change the data type or the device of the data
    """

    backend: str = "numpy"
    dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self):
        super().__post_init__()

        if self.backend == "numpy":
            if self.device != "cpu":
                raise ValueError("Converting to numpy only supports device as cpu")
        elif self.backend == "torch":
            if self.dtype == "float64":
                self.dtype = torch.float64
            elif self.dtype == "float32":
                self.dtype = torch.float32
            elif self.dtype == "float16":
                self.dtype = torch.float16
            else:
                raise ValueError(
                    "Supported torch data types: float64, float32, float16"
                )
        else:
            raise ValueError("Supported backends: 'numpy' or 'torch'")

    def transform(self, pad):
        frame = self.preparedframes[self.sink_pads[0]]

        outbufs = []
        for buf in frame:
            if buf.is_gap:
                out = None
            else:
                data = buf.data
                if self.backend == "numpy":
                    if isinstance(data, np.ndarray):
                        # numpy to numpy
                        out = data.astype(self.dtype, copy=False)
                    elif isinstance(data, torch.Tensor):
                        # torch to numpy
                        out = data.detach().cpu().numpy().astype(self.dtype, copy=False)
                    else:
                        raise ValueError("Unsupported data type")
                else:
                    if isinstance(data, np.ndarray):
                        # numpy to torch
                        out = torch.from_numpy(data).to(self.dtype).to(self.device)
                    elif isinstance(data, torch.Tensor):
                        # torch to torch
                        out = data.to(self.dtype).to(self.device)
                    else:
                        raise ValueError("Unsupported data type")

            outbufs.append(
                SeriesBuffer(
                    offset=buf.offset,
                    sample_rate=buf.sample_rate,
                    data=out,
                    shape=buf.shape,
                )
            )

        return TSFrame(
            buffers=outbufs,
            metadata=frame.metadata,
            EOS=frame.EOS,
        )
