import random
import torch


class RandomShift(object):
    """
    Shift features a bit in all coordinates
    """

    def __init__(self, low=0.95, high=1.05, prob=0.5, mask=None):
        assert isinstance(mask, (type(None), torch.Tensor))
        self._low = low
        self._high = high
        self._prob = prob
        self._mask = mask

    def __call__(self, data):
        if random.random() > self._prob:
            tr = (self._low - self._high) * torch.randn_like(
                data["x"]
            ) + self._high
            if self._mask is not None:
                tr[self._mask] = 1
            data["x"] = data["x"] * tr

        return data
