import argparse
import glob
import os
from typing import Dict, List

import numpy as np
from PIL import Image
from natsort import natsorted
from skimage.morphology import isotropic_erosion
import cc3d
import csv
from skimage.segmentation import watershed


def seeded_watershed(mask: np.array, erosion=True, erosion_radius=None, ccl_conn=6, seed_size=10) -> np.ndarray:
    """
    Seeded watershed.

    Args:
        mask: 3D binary mask.
        erosion: Apply erosion to the mask or not.
        erosion_radius: Control the extent of erosion.
        ccl_conn: Control the connectivity of the ccl kernel.
        seed_size: Control the size of the seeds.

    Returns:
        Instance segmentation result.

    """
    # erosion
    if erosion:
        if erosion_radius is None:
            raise ValueError("erosion_radius must be specified if erosion is True")
        sure_fg = isotropic_erosion(mask, erosion_radius)
    else:
        sure_fg = mask

    # CCL - create seeds
    seeds = cc3d.connected_components(sure_fg, connectivity=ccl_conn)

    # remove small seeds
    unique_labels = np.unique(seeds)
    unique_labels = unique_labels[unique_labels != 0]
    for label in unique_labels:
        if len(np.argwhere(seeds == label)) < seed_size:
            seeds[seeds == label] = 0

    # seeded watershed
    watershed_result = watershed(mask, seeds, mask=mask)

    return watershed_result


def save_csv(file_name: str, num_centroids_dict: Dict[str, int], volumes_dict: Dict[str, List[int]],
             centroids_dict: Dict[str, List[List[float]]]):
    """
    Save chamber information, including the number of chambers, volumes and centroids.

    Args:
        file_name: Name of the output CSV file.
        num_centroids_dict: Dictionary mapping chamber names to the number of centroids.
        volumes_dict: Dictionary mapping chamber names to their volume.
        centroids_dict: Dictionary mapping chamber names to a list of centroid coordinates.

    """

    header = ["name", "num_centroids", "volumes"] + [str(i) for i in range(1, len(num_centroids_dict) + 1)]
    with open(file_name, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for key, value in centroids_dict.items():
            row = [key] + [str(num_centroids_dict[key])] + [str(volumes_dict[key])] + [str(lst) for lst in
                                                                                       centroids_dict[key]]
            writer.writerow(row)


def save_sw(sample_name: str, watershed_result: np.ndarray):
    """
    Save instance segmentation results.

    Args:
        sample_name: Name of the sample.
        watershed_result: Instance segmentation result.

    """

    if not os.path.exists(os.path.join(args.save_img_pth, sample_name)):
        os.makedirs(os.path.join(args.save_img_pth, sample_name))

    # Save slice by slice (0.png, 1.png, ...)
    for i in range(watershed_result.shape[2]):
        mask_img = Image.fromarray(watershed_result[:, :, i].astype(np.uint8))
        mask_img.save(os.path.join(args.save_img_pth, os.path.join(sample_name, str(i) + ".png")))


def stack(pth: str) -> np.ndarray:
    """
    Stack 2D slices to form 3D data.

    Args:
        pth: Directory of 2D slices.

    Returns:
        3D data.

    """

    pths = natsorted(glob.glob(os.path.join(pth, "*")))
    mask = []
    for pth in pths:
        slice = np.array(Image.open(pth))
        mask.append(slice)
    mask = np.stack(mask, axis=0)
    mask = np.transpose(mask, (1, 2, 0))  # (x, y, z) format

    return mask


def run():
    sample_names = os.listdir(args.root)

    num_centroids_dict = {}
    volumes_dict = {}
    centroids_dict = {}

    for sample_name in sample_names:
        # stack slices to form the 3d data
        mask = stack(os.path.join(args.root, sample_name))

        # perform seeded watershed on the 3d data
        watershed_result = seeded_watershed(mask, erosion=True, erosion_radius=3, seed_size=10)

        # calculate chamber geo info: num. of chambers, volumes per chamber, centroids per chamber
        unique_labels = np.unique(watershed_result)
        unique_labels = unique_labels[unique_labels != 0]
        volumes = []
        centroids = []
        for label in unique_labels:
            label_indices = np.argwhere(watershed_result == label)
            volume = len(label_indices)
            volumes.append(volume)
            centroid = np.mean(label_indices, axis=0).astype(int)
            centroids.append(centroid.tolist())

        num_centroids_dict[sample_name] = len(volumes)
        volumes_dict[sample_name] = volumes
        centroids_dict[sample_name] = centroids

        # save instance segmentation results
        if args.save_img:
            print("Saving images...")
            save_sw(sample_name, watershed_result)

        print(sample_name)
        print(f"Number of chambers: {num_centroids_dict[sample_name]}")
        print(f"Volumes: {volumes_dict[sample_name]}")
        print(f"Centroids: {centroids_dict[sample_name]}")
        print("\n")

    # save chamber information to the CSV file
    if args.save_csv:
        print("Saving CSV file...")
        save_csv(args.save_csv_pth, num_centroids_dict, volumes_dict, centroids_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Seeded watershed on binary masks")
    parser.add_argument("--root", type=str, default="./",
                        help="Path of all semantic segmentation results divided by samples")
    parser.add_argument("--save-csv", type=bool, default=False,
                        help="Save chamber information to the csv file")
    parser.add_argument("--save-csv-pth", type=str, default="./chambers_geo.csv",
                        help="File name of csv")
    parser.add_argument("--save-img", type=bool, default=False,
                        help="Save instance segmentation results")
    parser.add_argument("--save-img-pth", type=str, default="./",
                        help="Path to save the instance segmentation results")
    args = parser.parse_args()
    run()