import os
import glob

import torch
from torch.utils.data import Dataset
import numpy as np

from xmlPAGE import pageData
from utils import mkdir, files_exist


def text_line_features_from_xml(xml_file, categories):
    """
    """
    page = pageData(xml_file)
    page.parse()
    img_size = page.get_size()
    text_regions = page.get_sorted_child("TextRegion")
    x = []
    sorted_lines = []

    if text_regions != None:
        for region in text_regions:
            rid = page.get_id(region)
            text_lines = page.get_sorted_child("TextLine", region)
            if text_lines != None:
                for line in text_lines:
                    data = {}
                    cl = torch.zeros(len(categories), dtype=torch.float)
                    line_id = page.get_id(line)
                    data["id"] = line_id
                    data["parent"] = page.name
                    line_type = page.get_region_type(line)
                    if line_type == None:
                        # -- use parent type
                        print(
                            "Type missing at {} {}, searching for parent type".format(
                                page.name, line_id
                            )
                        )
                        line_type = page.get_region_type(region)
                    if line_type == None:
                        print(
                            "Type search fail, using region name instead: {}".format(
                                "TextRegion"
                            )
                        )
                        line_type = "TextRegion"
                    cl[categories.index(line_type)] = 1
                    line_coords = page.get_baseline(line) / img_size
                    line_center = np.mean(line_coords, axis=0)
                    data["features"] = torch.cat(
                        (
                            cl,  # one hot encoded line type
                            torch.tensor(
                                line_center, dtype=torch.float
                            ),  # line center
                            torch.tensor(
                                line_coords[0, :], dtype=torch.float
                            ),  # start_point coord
                            torch.tensor(
                                line_coords[-1, :], dtype=torch.float
                            ),  # end point coord
                        )
                    )
                    # line_length = np.array([
                    #    line_coords[:,0].max()-line_coords[:,0].min(),
                    #    line_coords[:,1].max()-line_coords[:,1].min()
                    #    ])
                    x.append(data)
                    sorted_lines.append(line_id)

    table_regions = page.get_sorted_child("TableRegion")
    if table_regions != None:
        for region in table_regions:
            rid = page.get_id(region)
            cells = page.get_sorted_child("TableCell", region)
            for cell in cells:
                text_lines = page.get_sorted_child("TextLine", cell)
                if text_lines != None:
                    for line in text_lines:
                        data = {}
                        cl = torch.zeros(len(categories), dtype=torch.float)
                        line_id = page.get_id(line)
                        data["id"] = line_id
                        data["parent"] = page.name
                        line_type = page.get_region_type(line)
                        if line_type == None:
                            # -- use parent type
                            print(
                                "Type missing at {} {}, searching for parent type".format(
                                    page.name, line_id
                                )
                            )
                            line_type = page.get_region_type(region)
                        if line_type == None:
                            print(
                                "Type search fail, using region name instead: {}".format(
                                    "TableRegion"
                                )
                            )
                            line_type = "TextRegion"
                        cl[categories.index(line_type)] = 1
                        line_coords = page.get_baseline(line) / img_size
                        line_center = np.mean(line_coords, axis=0)
                        data["features"] = torch.cat(
                            (
                                cl,  # one hot encoded line type
                                torch.tensor(
                                    line_center, dtype=torch.float
                                ),  # line center
                                torch.tensor(
                                    line_coords[0, :], dtype=torch.float
                                ),  # start_point coord
                                torch.tensor(
                                    line_coords[-1, :], dtype=torch.float
                                ),  # end point coord
                            )
                        )
                        x.append(data)
                        sorted_lines.append(line_id)
    if len(sorted_lines) == 0:
        return None
    else:
        return (x, sorted_lines, page.name)


class TextLineInMemoryDataset(Dataset):
    """
    Basic Text-line based dataset 
    """

    def __init__(
        self,
        raw_data,
        set_id="train",
        processed_data="./processed",
        categories=["page_number", "paragraph", "marginalia"],
        transform=None,
        force_regenerate=False,
        soft_val=False,
    ):
        assert isinstance(raw_data, str)
        assert set_id in ["train", "val", "test", "prod"]
        assert isinstance(processed_data, str)
        assert isinstance(categories, list)
        super(TextLineInMemoryDataset, self).__init__()
        self._RAW_EXTENSION = "xml"
        self._PROCESSED_EXTENSION = "pickle"
        self._raw_data = raw_data
        self._set_id = set_id
        self._processed_data = processed_data
        self._categories = categories
        self._transform = transform
        self._filenames = self.raw_filenames()
        self._soft_val = soft_val
        self._processed_file = os.path.join(
            self._processed_data, self._set_id + "." + self._PROCESSED_EXTENSION
        )
        # self._processed_files = [os.path.join(self._processed_data,
        #    x + self._PROCESSED_EXTENSION) for x in self._filenames]

        if files_exist([self._processed_file]) and force_regenerate == False:
            print("Loading pre-processed {} data...".format(self._set_id))
            self.data, self.relatives, self.order = torch.load(
                self._processed_file
            )
            print("Done loading.")
        else:
            print("Processig {} data...".format(self._set_id))
            self.pre_process()
            print("Done processing.")
        if self._set_id in ["val", "test", "prod"]:
            # --- for val, test, prod build all posible pairs instead of generate
            # ---    them randomly as in 'train'
            if self._set_id == 'val' and self._soft_val == True:
                pass
            else:
                self._build_pairs()

    def raw_filenames(self):
        return [
            x
            for x in glob.glob(
                os.path.join(self._raw_data, "*." + self._RAW_EXTENSION)
            )
        ]

    def get_num_features(self):
        # --- one-hot category + center + start_point + end_point
        return 2 * (len(self._categories) + 2 + 2 + 2)

    def pre_process(self):
        # --- make out dir
        mkdir(self._processed_data)

        self._processed_files = []
        data_list = []
        data_relatives = {}
        data_order = {}
        idx = 0
        for f in self._filenames:
            # file_path = os.path.join(self._raw_data, f + "." + self._RAW_EXTENSION)
            page_data = text_line_features_from_xml(f, self._categories)
            if page_data:
                self._processed_files.append(f)
                # data = {"features":[],"id":[],"parent":[],"relationships":{}}
                data_relatives[page_data[2]] = []
                data_order[page_data[2]] = page_data[1]
                for j, data in enumerate(page_data[0]):
                    data_relatives[page_data[2]].append(idx)
                    data_list.append(data)
                    idx += 1
            else:
                print(
                    "File {} contains no data. Droped from {} set.".format(
                        file_path, self._set_id
                    )
                )
        torch.save(
            (data_list, data_relatives, data_order), self._processed_file
        )
        self.data = data_list
        self.relatives = data_relatives
        self.order = data_order

    def __getitem__(self, idx):
        if self._set_id == "train" or (
            self._set_id == "val" and self._soft_val == True
        ):
            # --- for each time a sample is selected gen a random pair from its
            # ---    relatives
            relatives = self.relatives[self.data[idx]["parent"]]
            ridx = torch.randint(0, len(relatives) - 1, (1,)).item()
            while relatives[ridx] == idx:
                ridx = torch.randint(0, len(relatives) - 1, (1,)).item()
            ridx = relatives[ridx]

            x = torch.cat(
                (self.data[idx]["features"], self.data[ridx]["features"])
            )
            z = (self.data[idx]["id"], self.data[ridx]["id"])
            # print(
            #        self.data[idx]['parent'], self.data[idx]['id'],
            #        " vs ",
            #        self.data[ridx]['parent'], self.data[ridx]['id']
            #        )
            y = 0 if idx >= ridx else 1
            y = torch.tensor(y, dtype=torch.float)

            sample = {"x": x, "t": y, "z": z}
            if self._transform:
                sample = self._transform(sample)

        elif self._set_id == "val" and self._soft_val == False:
            sample = self.pairs[idx]
        elif self._set_id == "test":
            sample = self.pairs[idx]
        elif self._set_id == "prod":
            sample = self.pairs[idx]

        return sample

    def _build_pairs(self):
        print("Build pairs for {} set".format(self._set_id))
        pairs = []
        for parent, childs in self.relatives.items():
            for i in childs:
                for j in childs:
                    if i == j:
                        # --- ignore self to self comp since the results is know
                        continue
                    x = torch.cat(
                        (self.data[i]["features"], self.data[j]["features"])
                    )
                    y = 0 if i >= j else 1
                    y = torch.tensor(y, dtype=torch.float)
                    z = (self.data[i]["id"], self.data[j]["id"])
                    pairs.append({"x": x, "t": y, "z": z, "parent": parent})
        self.pairs = pairs

    def __len__(self):
        if self._set_id in ["test", "prod"]:
            return len(self.pairs)
        elif self._set_id == "val" and self._soft_val == False:
            return len(self.pairs)
        elif self._set_id == "val" and self._soft_val == True:
            return len(self.data)
        return len(self.data)
