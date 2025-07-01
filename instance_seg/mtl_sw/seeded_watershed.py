import argparse
import os
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import isotropic_erosion
import cc3d
from skimage.segmentation import watershed
from utils import stack_imgs, save_chamber_info, save_slice_by_slice


def seeded_watershed(mask: np.ndarray, erosion: bool = True, erosion_radius: Optional[int] = None, ccl_conn: int = 6,
                     seed_size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
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
        if len(np.argwhere(seeds == label)) <= seed_size:
            seeds[seeds == label] = 0

    # seeded watershed
    watershed_result = watershed(mask, seeds, mask=mask)

    return watershed_result, seeds


def run():
    sample_names = os.listdir(args.root)

    num_centroids_dict = {}
    volumes_dict = {}
    centroids_dict = {}

    for sample_name in sample_names:
        print(sample_name)

        # stack slices to form the 3d data
        mask = stack_imgs(os.path.join(args.root, sample_name))

        # smooth
        mask = gaussian_filter(mask.astype(float), sigma=1)
        mask = mask > 0.5

        # perform seeded watershed on the 3d data
        watershed_result, seeds = seeded_watershed(mask, erosion=False, seed_size=10)

        # calculate chamber geo info: num. of chambers, volumes per chamber, centroids per chamber
        unique_labels = np.unique(seeds)
        unique_labels = unique_labels[unique_labels != 0]
        volumes = []
        centroids = []
        for label in unique_labels:
            label_indices = np.argwhere(seeds == label)
            volume = len(label_indices)
            volumes.append(volume)
            centroid = np.mean(label_indices, axis=0)
            centroids.append(centroid.tolist())

        num_centroids_dict[sample_name] = len(volumes)
        volumes_dict[sample_name] = volumes
        centroids_dict[sample_name] = centroids

        # save instance segmentation results
        if args.save_img:
            print("Saving images...")
            save_slice_by_slice(os.path.join(args.save_img_pth, sample_name), watershed_result, dim=2, format=".png")
            print("Images saved.")

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
    parser = argparse.ArgumentParser(description="Seeded watershed on binary masks")
    parser.add_argument("--root", type=str, default=".",
                        help="Path of all semantic segmentation results divided by samples")
    parser.add_argument("--save-csv", type=bool, default=True,
                        help="Save chamber information to the csv file")
    parser.add_argument("--save-csv-pth", type=str, default="./chambers_geo.csv",
                        help="Path to save csv file")
    parser.add_argument("--save-img", type=bool, default=True,
                        help="Save instance segmentation results")
    parser.add_argument("--save-img-pth", type=str, default=".",
                        help="Path to save the instance segmentation results")
    args = parser.parse_args()
    run()
