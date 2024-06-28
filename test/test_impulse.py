from sgnligo.cbc.sort_bank import group_and_read_banks, SortedBank
import torch

from sgnts.sources import FakeSeriesSource

# Read in the svd banks!
svd_bank = ["H1-0250_GSTLAL_SVD_BANK_half-0-0.xml.gz",]
nbank_pretend = 0
nslice = -1
verbose = True

copy_block = 1
device = "cpu"
dtype = torch.float32

banks = group_and_read_banks(
    svd_bank=svd_bank, nbank_pretend=nbank_pretend, nslice=nslice, verbose=verbose
)

sorted_bank = SortedBank(
    banks=banks,
    copy_block=copy_block,
    device=device,
    dtype=dtype,
    nbank_pretend=nbank_pretend,
    nslice=nslice,
    verbose=verbose,
)

# Build pipeline

