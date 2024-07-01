from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..base import SeriesBuffer, TransformElement, TSFrame
from sgnts.transforms import Matmul


@dataclass
class TorchMatmul(Matmul):
    """
    Performs matrix multiplication with provided matrix.

    If a pad receives more then one buffer, matmul will be performed
    on the list of buffers one by one. The source pad will also output
    a list of buffers.

    Parameters:
    -----------
    matrix: Sequence[Any]
        the matrix to multiply the data with, out = matrix x data

    Assumptions:
    ------------
    - There is only one sink pad and one source pad
    """

    def matmul(self, a, b):
        return torch.matmul(a, b)
