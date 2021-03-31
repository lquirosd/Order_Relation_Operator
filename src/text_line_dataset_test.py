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


from text_line_dataset import PairsInMemoryDataset


class PairsLineInMemoryDataset(unittest.TestCase):
    def test_lines_process(self):
        # --- set random vars
        torch.manual_seed(8)
        #raw_path = "/data/DATASETS/OHG/work/OHG/data/b012/page/"
        raw_path = "/home/lquirosd/WORK/PhD/SortReadingOrder/data/val/"
        categories = ["$pag", "$tip", "$par", "$not", "$nop", "$pac"]
        dataset = PairsInMemoryDataset(
            raw_path,
            set_id='test',
            processed_data="./processed",
            categories=categories,
            force_regenerate=True,
            hierarchical=False,
            soft_val=False,
        )
        print("Len: ", len(dataset))
        print(dataset.get_num_features())
        loader = DataLoader(dataset, batch_size=1)
        #for i, s in enumerate(loader):
        #    pass
    def test_hier_lines(self):
        # --- set random vars
        torch.manual_seed(8)
        #raw_path = "/data/DATASETS/OHG/work/OHG/data/b012/page/"
        raw_path = "/home/lquirosd/WORK/PhD/SortReadingOrder/data/val/"
        categories = ["$pag", "$tip", "$par", "$not", "$nop", "$pac"]
        dataset = PairsInMemoryDataset(
            raw_path,
            set_id='test',
            processed_data="./processed",
            categories=categories,
            force_regenerate=True,
            hierarchical=True,
            soft_val=False,
        )
        print("Len: ", len(dataset))
        print(dataset.get_num_features())
        loader = DataLoader(dataset, batch_size=1)
        #for i, s in enumerate(loader):
        #    pass
    def test_regions_process(self):
        # --- set random vars
        torch.manual_seed(8)
        raw_path = "/data/DATASETS/OHG/work/OHG/data/b012/page/"
        categories = ["$pag", "$tip", "$par", "$not", "$nop", "$pac"]
        dataset = PairsInMemoryDataset(
            raw_path,
            set_id='train',
            processed_data="./processed",
            categories=categories,
            force_regenerate=True,
            hierarchical=False,
            level='region',
        )
        print("Len: ", len(dataset))
        print(dataset.get_num_features())
        loader = DataLoader(dataset, batch_size=1)
        for i, s in enumerate(loader):
            pass
