import os
import glob

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

from xmlPAGE import pageData
from utils import mkdir, files_exist

def region_features_from_xml(xml_file, categories):
    """
    """
    page = pageData(xml_file)
    page.parse()
    img_size = page.get_size()
    rtxt = page.get_sorted_child('TextRegion')
    ttxt = page.get_sorted_child('TableRegion')
    regions = []
    if rtxt is not None:
        regions.extend(rtxt)
    if ttxt is not None:
        regions.extend(ttxt)
    x=[]
    sorted_regions = []
    if len(regions) > 0:
        for region in regions:
            data = {}
            data["id"] = page.get_id(region)
            data["parent"] = page.name
            region_type = page.get_region_type(region)
            if region_type == None:
                region_tag = page.get_tag(region)
                if region_tag in categories:
                    region_type = region_tag
                else:
                    print ("Region type undefined for region {}, this region will be ignored.".format(data["id"]))
                    continue
            #--- features: -region type one hot encoding + fst 3 spatial moments + fst 3 central moments
            cl = torch.zeros(len(categories), dtype=torch.float)
            cl[categories.index(region_type)] = 1
            coords = page.get_coords(region) / img_size
            #coords = page.get_coords(region)
            minx = coords[:,0].min()
            maxx = coords[:,0].max()
            miny = coords[:,1].min()
            maxy = coords[:,1].max()
            m = cv2.moments(coords)
            #--- 
            data["features"] = torch.cat(
                (
                    cl,  # one hot encoded region type
                    torch.tensor(
                        #[m["m00"], m["m10"]/m["m00"], m['m01']/m["m00"]], dtype=torch.float
                        [m["m00"], m["m10"]/m["m00"], m['m01']/m["m00"]], dtype=torch.float
                    ),  # spatial moments
                    torch.tensor(
                        #[m['mu11'], m['mu20'], m['m02']], dtype=torch.float
                        [minx, maxx, miny, maxy], dtype=torch.float
                    ),  # central moments
                )
            )
            x.append(data)
            sorted_regions.append(data["id"])
    if len(sorted_regions) == 0:
        return None
    else:
        return (x, sorted_regions, page.name)

            
    

def text_line_features_from_xml(xml_file, categories, hier=False):
    """
    """
    page = pageData(xml_file)
    page.parse()
    img_size = page.get_size()
    text_regions = page.get_sorted_child("TextRegion")
    x = []
    sorted_lines = []
    l0_sorted_lines = []
    l1_sorted_lines = {}

    if text_regions != None:
        for region in text_regions:
            rid = page.get_id(region)
            l1_sorted_lines[rid+"_"+page.name] = []
            text_lines = page.get_sorted_child("TextLine", region)
            if text_lines != None:
                for line in text_lines:
                    data = {}
                    cl = torch.zeros(len(categories), dtype=torch.float)
                    line_id = page.get_id(line)
                    data["id"] = line_id
                    data["parent"] = page.name
                    data["l0_parent"] = rid
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
                    l0_sorted_lines.append(line_id)
                    l1_sorted_lines[rid+"_"+page.name].append(line_id)

    table_regions = page.get_sorted_child("TableRegion")
    if table_regions != None:
        for region in table_regions:
            rid = page.get_id(region)
            l1_sorted_lines[rid+"_"+page.name] = []
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
                        data["l0_parent"] = rid
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
                            line_type = "TableRegion"
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
                        l0_sorted_lines.append(line_id)
                        l1_sorted_lines[rid+"_"+page.name].append(line_id)
    if len(sorted_lines) == 0:
        return None
    else:
        return (x, sorted_lines, page.name, l1_sorted_lines)


class PairsInMemoryDataset(Dataset):
    """
    Basic Text-line based dataset 
    """

    def __init__(
        self,
        raw_data,
        set_id="train",
        processed_data="./processed",
        categories=["page_number", "paragraph", "marginalia"],
        hierarchical=False,
        transform=None,
        force_regenerate=False,
        soft_val=False,
        level='line',
    ):
        assert isinstance(raw_data, str)
        assert set_id in ["train", "val", "test", "prod"]
        assert isinstance(processed_data, str)
        assert isinstance(categories, list)
        assert level in ['line', 'region']
        super(PairsInMemoryDataset, self).__init__()
        self._RAW_EXTENSION             = "xml"
        self._PROCESSED_EXTENSION       = "pickle"
        self._raw_data                  = raw_data
        self._set_id                    = set_id
        self._processed_data            = processed_data
        self._categories                = categories
        self._hierarchical              = hierarchical
        self._transform                 = transform
        self._filenames                 = self.raw_filenames()
        self._force_regenerate          = force_regenerate
        self._soft_val                  = soft_val
        self._level                     = level

        self._processed_file = os.path.join(
            self._processed_data, self._set_id + "_" + self._level + "." + self._PROCESSED_EXTENSION
        )
        self.get_data()
        # self._processed_files = [os.path.join(self._processed_data,
        #    x + self._PROCESSED_EXTENSION) for x in self._filenames]

    def get_data(self):
        if files_exist([self._processed_file]) and self._force_regenerate == False:
            print("Loading pre-processed {} data...".format(self._set_id))
            self.data, self.relatives, self.order, hier, set_id = torch.load(
                self._processed_file
            )
            self._num_features = self.data[0]['features'].size()[0]
            print("Done loading.")
            if hier != self._hierarchical or set_id != self._set_id:
                print("Loaded data metadata differs to current specs, regenerating data...")
                self._force_regenerate = True
                self.get_data()
            else:
                self._build_pairs()
        else:
            print("Processig {} data...".format(self._set_id))
            self.pre_process()
            self._build_pairs()
            print("Done processing.")
        #if self._set_id in ["val", "test", "prod"]:
        #    # --- for val, test, prod build all posible pairs instead of generate
        #    # ---    them randomly as in 'train'
        #    if self._set_id == 'val' and self._soft_val == True:
        #        pass
        #    else:
        #        self._build_pairs()

    def raw_filenames(self):
        return [
            x
            for x in glob.glob(
                os.path.join(self._raw_data, "*." + self._RAW_EXTENSION)
            )
        ]

    def get_num_features(self):
        # --- one-hot category + center + start_point + end_point
        #return 2 * (len(self._categories) + 2 + 2 + 2)
        return 2*self._num_features

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
            if self._level == 'line':
                page_data = text_line_features_from_xml(f, self._categories)
            elif self._level == 'region':
                page_data = region_features_from_xml(f, self._categories)
            if page_data:
                self._processed_files.append(f)
                # data = {"features":[],"id":[],"parent":[],"relationships":{}}
                #data_relatives[page_data[2]] = []
                #if self._hierarchical:
                #    for data in page_data[0]:
                #        p = data["l0_parent"]+"_"+data["parent"]
                #        data_order[p] = page_data[3][p]
                #        if p not in data_relatives:
                #            data_relatives[p] = []
                #        data_relatives[p].append(idx)
                #        data_list.append(data)
                if self._hierarchical == False:
                    data_order[page_data[2]] = page_data[1]
                for data in page_data[0]:
                    if self._hierarchical:
                        p = data["l0_parent"]+"_"+data["parent"]
                    else:
                        p = data["parent"]
                    if p not in data_relatives:
                        data_relatives[p] = []
                    if self._hierarchical:
                        data_order[p] = page_data[3][p]
                    data_relatives[p].append(idx)
                    #data_relatives[page_data[2]].append(idx)
                    data_list.append(data)
                    idx += 1
            else:
                print(
                    "File {} contains no data. Droped from {} set.".format(
                        file_path, self._set_id
                    )
                )

        torch.save(
            (data_list, data_relatives, data_order, self._hierarchical, self._set_id), self._processed_file
        )
        self.data = data_list
        self.relatives = data_relatives
        self.order = data_order
        self._num_features = data_list[0]['features'].size()[0]

    def __getitem__(self, idx):
        if self._set_id == "train" or (
            self._set_id == "val" and self._soft_val == True
        ):
            # --- for each time a sample is selected gen a random pair from its
            # ---    relatives
            if self._hierarchical:
                p = self.data[idx]["l0_parent"] + "_" + self.data[idx]["parent"] 
            else:
                p = self.data[idx]["parent"]
            relatives = self.relatives[p]
            #--- if relativs == 1 means the element is alone, so the decoder will take care of it. But to keep dataloader to return samples on the same order a dummy pair is generated
            if len(relatives) == 1:
                ridx = idx
            else:
                ridx = torch.randint(0, len(relatives), (1,)).item()
                while relatives[ridx] == idx:
                    ridx = torch.randint(0, len(relatives), (1,)).item()
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

            sample = {"x": x, "t": y, "z": z, 'parent': p}
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
        if self._set_id == 'train' or (self._set_id == 'val' and self._soft_val == True):
            return None
        print("Build pairs for {} set".format(self._set_id))
        pairs = []
        for parent, childs in self.relatives.items():
            for i in childs:
                for j in childs:
                    if i == j and len(childs) != 1:
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


