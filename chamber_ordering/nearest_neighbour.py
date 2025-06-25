import argparse
import ast
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from utils import plot_trajectory


def nearest_neighbor_tsp(points: np.ndarray, start_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
        Solves a variation of the Traveling Salesman Problem (TSP) using the
        nearest neighbor heuristic.

        Starts from a given point and iteratively visits the nearest unvisited point
        until all points have been visited.

        Args:
            points: A 2D array of shape (N, D) where N is the number of points
                                 and D is the dimensionality (e.g., 2 for 2D, 3 for 3D).
            start_idx: The index of the starting point in the `points` array.

        Returns:
            np.ndarray: A (N, D) array representing the sequence of points visited.
            np.ndarray: A (N,) array of indices indicating the visiting order of original points.

        """
    # Start from the first point
    start_point = points[start_idx]
    remaining_points = np.delete(points, start_idx, axis=0)

    # Initialize the path with the first point
    path = [start_point]
    order = [start_idx]

    # While there are points left to visit
    while remaining_points.shape[0] > 0:  # Check if there are any points remaining
        # Compute the distance from the last point in the path to all remaining points
        distances = cdist([path[-1]], remaining_points)[0]

        # Find the index of the closest point
        nearest_point_idx = np.argmin(distances)

        # Add the closest point to the path
        path.append(remaining_points[nearest_point_idx])
        order.append(np.where((points == remaining_points[nearest_point_idx]).all(axis=1))[0][0])
        # Remove the visited point from the remaining points
        remaining_points = np.delete(remaining_points, nearest_point_idx, axis=0)

    return np.array(path), np.array(order)


def run():
    # the chamber info file obtained from instance seg
    df = pd.read_csv(args.csv_file)
    names = df["name"]

    for name in names:
        row = df.loc[df['name'] == name]
        pred_centroids = np.array(ast.literal_eval(row["centroids"].values[0]))
        pred_volumes = np.array(ast.literal_eval(row["volumes"].values[0]))

        # the trajectory starts with the chamber with the smallest volume
        start_idx = np.argmin(pred_volumes).item()

        # reconstruct the trajectory
        pred_path, _ = nearest_neighbor_tsp(pred_centroids, start_idx)

        # visualize the trajectory
        plot_trajectory(pred_path, pred_path, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reconstructing growth trajectory of chambers")
    parser.add_argument("--csv-file", type=str,
                        default=".",
                        help="Path to csv that stores the chamber information")

    args = parser.parse_args()
    run()
