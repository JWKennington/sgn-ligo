from dataclasses import dataclass

import torch
from sgnts.transforms import Resampler
from torch.nn.functional import conv1d as Fconv1d


@dataclass
class TorchResampler(Resampler):
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        super().__post_init__()
        if self.outrate < self.inrate:
            # downsample
            self.thiskernel = (
                torch.from_numpy(self.thiskernel)
                .view(1, 1, -1)
                .to(self.device)
                .to(self.dtype)
            )
        else:
            # upsample
            sub_kernel_length = int(2 * self.half_length + 1)
            self.thiskernel = (
                torch.tensor(self.thiskernel.copy())
                .view(self.outrate // self.inrate, 1, sub_kernel_length)
                .to(self.device)
                .to(self.dtype)
            )

    def resample(self, data0, output_shape):
        # FIXME: include memeory format
        data = data0.view(-1, 1, data0.shape[-1])
        thiskernel = self.thiskernel

        if self.outrate > self.inrate:  # upsample
            out = Fconv1d(data, thiskernel)
            out = out.mT.reshape(data.shape[0], -1)
        else:  # downsample
            out = Fconv1d(data, thiskernel, stride=self.inrate // self.outrate)
            out = out.squeeze(1)

        out = out.view(output_shape)

        return out
