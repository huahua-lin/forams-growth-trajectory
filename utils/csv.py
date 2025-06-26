import csv
from typing import Dict, List


def max_columns_in_csv(filepath: str) -> int:
    """
    Return the maximum number of columns used in a csv file.
    Args:
        filepath (str): Path to csv file

    Returns:
        int: Maximum number of columns in csv file

    """
    max_columns = 0
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            max_columns = max(max_columns, len(row))
    return max_columns


def save_chamber_info(file_name: str, num_centroids_dict: Dict[str, int], volumes_dict: Dict[str, List[int]],
                      centroids_dict: Dict[str, List[List[float]]]):
    """
    Save chamber information, including the number of chambers, volumes and centroids.

    Args:
        file_name: Name of the output CSV file.
        num_centroids_dict: Dictionary mapping chamber names to the number of centroids.
        volumes_dict: Dictionary mapping chamber names to their volume.
        centroids_dict: Dictionary mapping chamber names to a list of centroid coordinates.

    """
    header = ["name", "num_chambers", "volumes", "centroids"]
    with open(file_name, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for key, value in centroids_dict.items():
            row = [key] + [str(num_centroids_dict[key])] + [str(volumes_dict[key])] + [str(centroids_dict[key])]
            writer.writerow(row)
