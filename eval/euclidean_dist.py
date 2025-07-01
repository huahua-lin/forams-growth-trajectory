import argparse
import os
import ast
import numpy as np
import pandas as pd
from utils import stack_imgs, resize_volume_with_aspect_ratio


def run():
    df = pd.read_csv(args.pred_csv_pth)
    names = df["name"]
    avg_dist = []
    for name in names:
        gt_stack = stack_imgs(os.path.join(args.gt_seg_pth, name))
        gt_stack = resize_volume_with_aspect_ratio(gt_stack, (128, 128, 64))  # resize if needed

        # get pred centroids coordinates
        row = df.loc[df['name'] == name]
        pred_centroids = np.array(ast.literal_eval(row["centroids"].values[0]))

        dist_arr = {}
        for coord in pred_centroids:
            gt_label = gt_stack[int(coord[0]), int(coord[1]), int(coord[2])]
            # ground truth centroids coordinates
            gt_centroid = np.mean(np.argwhere(gt_stack == gt_label), axis=0)
            # Euclidean distance
            dist = np.linalg.norm(gt_centroid - coord)
            if gt_label in dist_arr:
                if dist < dist_arr[gt_label]:
                    dist_arr[gt_label] = dist
            else:
                dist_arr[gt_label] = dist
        dist_arr_values = dist_arr.values()
        avg_dist_per_sample = sum(dist_arr_values) / len(dist_arr_values)
        print(f"Average Euclidean distance between the pred centroids and the gt centroids for {name} is",
              avg_dist_per_sample)
        avg_dist.append(avg_dist_per_sample)
    print(avg_dist)
    print("Average Euclidean distance is", sum(avg_dist) / len(avg_dist))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reconstructing growth trajectory of chambers")
    parser.add_argument("--pred-csv-pth", type=str,
                        default=".",
                        help="A CSV file that stores the predicted chamber information")
    parser.add_argument("--gt-seg-pth", type=str,
                        default=".",
                        help="A directory that stores the ground truth instance segmentation results divided by samples")
    args = parser.parse_args()
    run()
