import unittest
import numpy as np
import torch
import torch.nn.functional as F
import os
import pickle
from torch.utils.data import DataLoader

try:
    import unittest.mock as mock
except ImportError:
    import mock


from text_line_dataset import TextLineInMemoryDataset


class TestTextLineInMemoryDataset(unittest.TestCase):
    def test_process(self):
        # --- set random vars
        torch.manual_seed(8)
        raw_path = "/data/DATASETS/OHG/work/OHG/data/b012/page/"
        categories = ["$pag", "$tip", "$par", "$not", "$nop", "$pac"]
        dataset = TextLineInMemoryDataset(
            raw_path,
            processed_data="./processed",
            categories=categories,
            force_regenerate=True,
        )
        print("Len: ", len(dataset))
        print(dataset.get_num_features())
        loader = DataLoader(dataset, batch_size=1)
        for i, s in enumerate(loader):
            pass
