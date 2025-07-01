import argparse
import ast
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from utils import stack_imgs
from chamber_ordering import nearest_neighbor_tsp
from utils import compare_trajectories, resize_volume_with_aspect_ratio


def run():
    df_pred = pd.read_csv(args.pred_csv_pth)
    names = df_pred["name"]
    spearman_list = []

    for name in names:
        gt_stack = stack_imgs(os.path.join(args.gt_seg_pth, name))
        # gt_stack = resize_volume_with_aspect_ratio(gt_stack, (128, 128, 64))  # resize if needed

        # get pred centroids coordinates and volumes
        row = df_pred.loc[df_pred['name'] == name]
        pred_centroids = np.array(ast.literal_eval(row["centroids"].values[0]))
        pred_volumes = np.array(ast.literal_eval(row["volumes"].values[0]))

        # pred the order, get the list of order
        start_idx = np.argmin(pred_volumes).item()
        pred_path, _ = nearest_neighbor_tsp(pred_centroids, start_idx)

        # get labels for each pred centroid in the gt labeled img
        gt_labels = []
        for idx, coord in enumerate(pred_path):
            label = gt_stack[int(coord[0]), int(coord[1]), int(coord[2])]
            if label != 0:
                gt_labels.append(
                    label)  # Some predicted centroids might be outside of chambers-> ground truth label == 0
            elif label == 0:
                pred_path = np.delete(pred_path, np.where((pred_path == coord).all(axis=1))[0], axis=0)
        gt_labels = np.array(gt_labels)

        sorted_lst = sorted(gt_labels)
        value_to_squeezed = {value: i for i, value in enumerate(sorted_lst)}
        gt_labels = np.array([value_to_squeezed[value] for value in gt_labels])

        # compare the gt labels with the pred order
        pred_order = np.arange(len(pred_path))
        rank = spearmanr(gt_labels, pred_order)[0]
        print(f"{name}: {rank}")

        if args.show:
            gt_path = pred_path[gt_labels]
            compare_trajectories(pred_centroids, pred_path, gt_path, name, rank)

        spearman_list.append(rank)

    print("Average spearman's rank:", sum(spearman_list) / len(spearman_list))
    print("Percentage of correct order:", sum(1 for x in spearman_list if x == 1.0) / len(spearman_list))
    print("Percentage of above 0.9:", sum(1 for x in spearman_list if x > 0.9) / len(spearman_list))
    print("Percentage of above 0.5:", sum(1 for x in spearman_list if x > 0.5) / len(spearman_list))
    print("Percentage of below 0:", sum(1 for x in spearman_list if x < 0) / len(spearman_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Comparing ground truth and prediction results (ordering)")
    parser.add_argument("--pred-csv-pth", type=str,
                        default=".",
                        help="File name of csv that stores the chamber information")
    parser.add_argument("--gt-seg-pth", type=str,
                        default=".",
                        help="Path that stores all the ground truth segmentation")
    parser.add_argument("--show", type=bool,
                        default=False, help="Show the trajectory comparison with the ground truth")
    args = parser.parse_args()
    run()
