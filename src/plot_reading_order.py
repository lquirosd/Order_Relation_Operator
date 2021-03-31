import os
import sys
import glob

import numpy as np
import pickle
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch


def main():
    out_dir = sys.argv[1]
    img_dir = sys.argv[2]
    gt_file = sys.argv[3]
    categories = sys.argv[4].split()
    color = sys.argv[5]
    try:
        hyp_dir = sys.argv[6]
    except:
        hyp_dir = None

    try:
        hyp2_dir = sys.argv[7]
    except:
        hyp2_dir = None

    if hyp_dir and hyp2_dir:
        hier = True
    else:
        hier = False

    c_map = {'color':color, 'marker':'x'}

    gt_data, gt_relatives, gt_order, _, _ = torch.load(gt_file)
    id_to_idx = {}
    for idx, d in enumerate(gt_data):
        id_to_idx[d["parent"] + d["id"]] = idx

    # img_path = []
    # img_names = []
    # for ext in ['tif', 'jpg','jpeg','png']:
    #    ext_paths = glob.glob(os.path.join(img_dir, "*."+ext))
    #    img_paths.append(ext_paths)
    #    img_names.append(
    #            [os.path.basename(x).strip("."+ext)for x in ext_paths]
    #            )
    c_pos = len(categories)
    for img_name in gt_order.keys():
        # --- TODO: change to any extension, no just tif
        for ext in ["tif", "jpg", "JPG", "png", "JPEG", "jpeg"]:
            if os.path.isfile(os.path.join(img_dir, img_name + "." + ext)):
                img_path = os.path.join(img_dir, img_name + "." + ext)
                break
            print("Image not found for file", img_name)
        img = mpimg.imread(img_path)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        z = gt_order[img_name]
        points = np.zeros((len(z), 2))
        for j, element in enumerate(z):
            feats = gt_data[id_to_idx[img_name + element]]["features"]
            center = np.array([feats[-2]+(feats[-1]-feats[-2])/2, feats[-4]+(feats[-3]-feats[-4])/2])
            points[j] = (center*(img.shape[0:2])).astype(np.int)[::-1] - np.array([30, 0])
            #points[j] = (
            #        gt_data[id_to_idx[img_name + element]]["features"][c_pos:c_pos+2].numpy()
            #    * (img.shape[0:2][::-1])
            #).astype(np.int) - np.array([30, 0])
            

        #plt.plot(points[:, 0], points[:, 1], "go-")
        plt.plot(points[:, 0], points[:, 1], 
                color='green', 
                marker='o', 
                markeredgecolor='black', 
                markerfacecolor='None',
                )
        if hyp_dir and not hyp2_dir:
            points = np.zeros((len(z), 2))
            with open(os.path.join(hyp_dir, img_name + ".pickle"), "rb") as fh:
                s = pickle.load(fh)[0]
            for j, element in enumerate(s):
                feats = gt_data[id_to_idx[img_name + z[element]]]["features"]
                center = np.array([feats[-2]+(feats[-1]-feats[-2])/2, feats[-4]+(feats[-3]-feats[-4])/2])
                points[j] = (center*(img.shape[0:2])).astype(np.int)[::-1] + np.array([30, 0])
                #points[j] = (
                #    gt_data[id_to_idx[img_name + z[element]]]["features"][
                #        c_pos:c_pos+2
                #        ].numpy()
                #    * (img.shape[0:2][::-1])
                #).astype(np.int) + np.array([30, 0])
            plt.plot(points[:, 0], points[:, 1], **c_map)

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            out_dir + "/" + img_name + ".svg", dpi=150, bbox_inches="tight"
        )
        plt.close()


if __name__ == "__main__":
    main()
