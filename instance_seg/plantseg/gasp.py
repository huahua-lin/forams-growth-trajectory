import argparse
import os

import numpy as np
from elf.segmentation import distance_transform_watershed
from multicut import gasp

from utils import stack_imgs, save_chamber_info, save_slice_by_slice


def run():
    sample_names = os.listdir(args.pred_pth)

    num_centroids_dict = {}
    volumes_dict = {}
    centroids_dict = {}

    for sample_name in sample_names:
        # stack slices to form the 3d data
        pred = stack_imgs(os.path.join(args.pred_pth, sample_name))
        mask = stack_imgs(os.path.join(args.mask_pth, sample_name))  # we use binary masks from 2D approach

        # probability map of boundaries
        pmaps = pred / 255

        # Watershed - generate supervoxels
        ws, _ = distance_transform_watershed(pmaps, mask=mask, threshold=0.3, sigma_seeds=1, min_size=10)

        # GASP
        seg = gasp(pmaps, ws, beta=0.8, post_minsize=10)

        # calculate chamber geo info: num. of chambers, volumes per chamber, centroids per chamber
        unique_labels = np.unique(seg)
        unique_labels = unique_labels[unique_labels != 0]
        volumes = []
        centroids = []
        for label in unique_labels:
            label_indices = np.argwhere(seg == label)
            volume = len(label_indices)
            if volume > 10 and volume < 200000:
                volumes.append(volume)
                centroid = np.mean(label_indices, axis=0)
                centroids.append(centroid.tolist())
            else:
                seg[seg == label] = 0

        num_centroids_dict[sample_name] = len(volumes)
        volumes_dict[sample_name] = volumes
        centroids_dict[sample_name] = centroids

        # save instance segmentation results
        if args.save_img:
            print("Saving images...")
            save_slice_by_slice(os.path.join(args.save_img_pth, sample_name), seg, dim=2, format=".png")
            print("Images saved.")

        print(sample_name)
        print(f"Number of chambers: {num_centroids_dict[sample_name]}")
        print(f"Volumes: {volumes_dict[sample_name]}")
        print(f"Centroids: {centroids_dict[sample_name]}")
        print("\n")

    # save chamber information to the CSV file
    if args.save_csv:
        print("Saving CSV file...")
        save_chamber_info(args.save_csv_pth, num_centroids_dict, volumes_dict, centroids_dict)
        print("CSV file saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GASP on binary masks")
    parser.add_argument("--pred-pth", type=str, default=".",
                        help="Path of all semantic segmentation results divided by samples")
    parser.add_argument("--mask-pth", type=str, default=".",
                        help="Path of all semantic segmentation results (mask) divided by samples")
    parser.add_argument("--save-csv", type=bool, default=False,
                        help="Save chamber information to the csv file")
    parser.add_argument("--save-csv-pth", type=str, default="./chambers_geo.csv",
                        help="Path to save csv file")
    parser.add_argument("--save-img", type=bool, default=True,
                        help="Save instance segmentation results")
    parser.add_argument("--save-img-pth", type=str, default=".",
                        help="Path to save the instance segmentation results")
    args = parser.parse_args()
    run()
