import os
import sys
import glob
import argparse
from tqdm import tqdm

import numpy as np
import pickle
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch


def plot_hier_data(out_dir, img_dir, gt_regions, gt_lines, hyp_regions, hyp_lines, c_pos, c_map, out_format):
    reg_data, reg_relatives, reg_order, _, _ = torch.load(gt_regions)
    lin_data, lin_relatives, lin_order, _, _ = torch.load(gt_lines)
    id_to_idx = {}
    for idx, d in enumerate(reg_data):
        id_to_idx[d["parent"] + d["id"]] = idx
    l_id_to_idx = {}
    for idx, d in enumerate(lin_data):
        l_id_to_idx[d["parent"] + d["l0_parent"] + d["id"]] = idx

    for img_name in tqdm(reg_order.keys()):
        img_path = None
        for ext in ["tif", "jpg", "JPG", "png", "JPEG", "jpeg"]: 
            if os.path.isfile(os.path.join(img_dir, img_name + "." + ext)):
                img_path = os.path.join(img_dir, img_name + "." + ext)
                break
        if img_path is None:
            print("Image not found for file", img_name)
            exit()
        img = mpimg.imread(img_path)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        reg_z = reg_order[img_name]
        points=[]
        for j, element in enumerate(reg_z):
            #--- get line level data
            if element + "_" + img_name in lin_order:
                lin_z = lin_order[element + "_" + img_name]
            else:
                continue
            for i, line in enumerate(lin_z):
                feats = lin_data[l_id_to_idx[img_name + element + line]]["features"]
                center = (feats[c_pos:c_pos+2].numpy()*img.shape[0:2][::-1]).astype(np.int) - np.array([30, 0])
                points.append(center)
        points = np.array(points)
        plt.plot(points[:, 0], points[:, 1], 
                color='green', 
                marker='o', 
                markeredgecolor='black', 
                markerfacecolor='None',
                )
        #---- check for hyps
        if hyp_regions and hyp_lines:
            points = []
            with open(os.path.join(hyp_regions, img_name + ".pickle"), "rb") as fh:
                reg_s = pickle.load(fh)[0]
            for j, element in enumerate(reg_s):
                region = reg_z[element]
                if not os.path.isfile(os.path.join(hyp_lines, region + "_" +img_name + ".pickle")):
                    print("file not found", os.path.join(hyp_lines, region + "_" +img_name + ".pickle"))
                    continue
                with open(os.path.join(hyp_lines, region + "_" +img_name + ".pickle"), "rb") as fh:
                    lin_s = pickle.load(fh)[0]
                for i, line in enumerate(lin_s):
                    l_id = lin_order[region + "_" + img_name][line]
                    feats = lin_data[l_id_to_idx[img_name + region + l_id]]["features"]
                    center = (feats[c_pos:c_pos+2].numpy()*img.shape[0:2][::-1]).astype(np.int) + np.array([30, 0])
                    points.append(center)
            points = np.array(points)
            #fst = c_map.copy()
            #fst['marker'] = 'D'
            #fst['color'] = 'red'
            plt.plot(points[:, 0], points[:, 1], **c_map)
            #plt.plot(points[0, 0], points[0, 1], **fst)

        
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            out_dir + "/" + img_name + "." + out_format, dpi=150, bbox_inches="tight"
        )
        plt.close()
            




def plot_page_data(out_dir, img_dir, gt, hyp, c_pos, c_map, out_format):
    gt_data, gt_relatives, gt_order, _, _ = torch.load(gt)
    id_to_idx = {}
    for idx, d in enumerate(gt_data):
        id_to_idx[d["parent"] + d["id"]] = idx

    for img_name in tqdm(gt_order.keys()):
        img_path = None
        for ext in ["tif", "jpg", "JPG", "png", "JPEG", "jpeg"]:
            if os.path.isfile(os.path.join(img_dir, img_name + "." + ext)):
                img_path = os.path.join(img_dir, img_name + "." + ext)
                break
        if img_path is None:
            print("Image not found for file", img_name)
            exit()
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
        if hyp:
            points = np.zeros((len(z), 2))
            with open(os.path.join(hyp, img_name + ".pickle"), "rb") as fh:
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
            #fst = c_map.copy()
            #fst['marker'] = 'D'
            #fst['color'] = 'red'
            #plt.plot(points[:, 0], points[:, 1], **c_map)
            #plt.plot(points[0, 0], points[0, 1], **fst)

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            out_dir + "/" + img_name + "." + out_format, dpi=150, bbox_inches="tight"
        )
        plt.close()




def main():
    parser = argparse.ArgumentParser(description='Plot reading orrder data')
    parser.add_argument('--img_dir', 
            type=str, 
            default='./',
            help='Pointer to base images to plot reading order over',
            )
    parser.add_argument('--out_dir', 
            type=str, 
            default='./',
            help='Pointer to base images to save plots',
            )
    parser.add_argument('--format', 
            type=str, 
            default='svg',
            choices=['svg', 'jpg'],
            help='format to save images',
            )
    parser.add_argument('--hierarchical', 
            action='store_true',
            help='Use hierarchical data',
            )
    parser.add_argument('--gt_lines', 
            type=str, 
            default=None,
            help='Pointer to gt lines file at page level',
            )
    parser.add_argument('--hyp_lines', 
            type=str, 
            default=None,
            help='Pointer to hypothesis dir for lines at page level',
            )
    parser.add_argument('--gt_regions', 
            type=str, 
            default=None,
            help='Pointer to gt regions at page level',
            )
    parser.add_argument('--hyp_regions', 
            type=str, 
            default=None,
            help='Pointer to hypothesis dir for regions at page level',
            )
    parser.add_argument('--gt_hier_lines', 
            type=str, 
            default=None,
            help='Pointer to gt lines file at region level',
            )
    parser.add_argument('--hyp_hier_lines', 
            type=str, 
            default=None,
            help='Pointer to hypothesis dir for lines at region level',
            )
    parser.add_argument('--color', 
            type=str, 
            default='blue',
            help='color to plot the hyp data.',
            )
    parser.add_argument('--num_categories', 
            type=int, 
            default=6,
            help='Number of categories used to gen the data',
            )

    args = parser.parse_args()

    plot_cmap = {'color':args.color, 'marker':'x'}

    if args.hierarchical:
        if None in [args.gt_regions,args.gt_hier_lines,args.hyp_regions,args.hyp_hier_lines]:
            print("Error: gt_regions, gt_hier_lines, hyp_regions and hyp_hier_lines must be defined for hierarchical plot.")
            exit()

        plot_hier_data(args.out_dir,args.img_dir, args.gt_regions, args.gt_hier_lines, 
                args.hyp_regions, args.hyp_hier_lines,
                args.num_categories, plot_cmap, args.format)
    else:
        if args.gt_lines is not None:
            plot_page_data(args.out_dir, args.img_dir, args.gt_lines, args.hyp_lines, 
                    args.num_categories, plot_cmap, args.format)
        elif args.gt_regions is not None:
            plot_page_data(args.out_dir, args.img_dir, args.gt_regions, args.hyp_regions, 
                    args.num_categories, plot_cmap, args.format)
        else:
            print("Error: Combination of inputs is not supported. Please provide gt_lines or gt_regions at non-hierarchical plot")
            exit()




if __name__=='__main__':
    main()
