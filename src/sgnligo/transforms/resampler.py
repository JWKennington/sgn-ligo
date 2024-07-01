from . import *
from ..base import *
from torch.nn.functional import conv1d as Fconv1d


@dataclass
class TorchResampler(Resampler):
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        super().__post_init__()
        if self.outrate < self.inrate:
            # downsample
            self.thiskernel = self.downkernel(self.factor)
        else:
            # upsample
            self.thiskernel = self.upkernel(self.factor)

    def resample(self, data0, output_shape):
        # FIXME: include memeory format
        data = data0.view(-1, 1, data0.shape[-1])
        factor = self.factor
        thiskernel = self.thiskernel

        if factor > 1:  # upsample
            out = Fconv1d(data, thiskernel)
            out = out.mT.reshape(data.shape[0], -1)
        else:  # downsample
            out = Fconv1d(data, thiskernel, stride=int(1 / factor))
            out = out.squeeze(1)

        out = out.view(output_shape)

        return out

    def downkernel(self, factor: float):
        """
        Compute the kernel for downsampling
        """
        kernel_length = int(2 * self.half_length + 1)

        # the domain should be the kernel_length divided by two
        c = kernel_length // 2
        x = torch.arange(-c, c + 1, device=self.device, dtype=self.dtype)
        vecs = torch.sinc(x * factor) * torch.sinc(x / c)
        norm = torch.norm(vecs) / factor**0.5
        vecs = vecs / norm

        return vecs.view(1, 1, -1)

    def upkernel(self, factor: float):
        """
        Compute the kernel for upsampling
        """
        factor = int(factor)

        kernel_length = int(2 * self.half_length * factor + 1)
        sub_kernel_length = int(2 * self.half_length + 1)

        # the domain should be the kernel_length divided by two
        c = kernel_length // 2
        x = torch.arange(-c, c + 1, device=self.device, dtype=self.dtype)
        out = torch.sinc(x / factor) * torch.sinc(x / c)
        # out = tpad(out, (0, factor - 1))
        out = self.adapter_config.lib.pad_func(out, (0, factor - 1))
        # FIXME: check if interleave same as no interleave
        vecs = out.reshape(-1, factor).T.flip(-1)

        return vecs.view(int(factor), 1, sub_kernel_length)
