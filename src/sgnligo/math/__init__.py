import torch
from torch.nn.functional import pad as tpad


def cat_func(xs, axis):
    return torch.cat(xs, dim=axis)


def pad_func(data, pad_samples):
    return tpad(data, pad_samples, "constant")


def full_func(shape, fill_value):
    return torch.full(shape, fill_value)


def stack_func(data, axis=0):
    return torch.stack(data, axis)


zeros_func = torch.zeros
