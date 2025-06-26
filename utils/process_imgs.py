import glob
import os

import numpy as np
from PIL import Image
from natsort import natsorted


def stack_imgs(pth: str, format='xyz') -> np.ndarray:
    """
    Stack 2D slices to form 3D data.

    Args:
        pth: Directory of 2D slices.
        format: Format of 3D volume.

    Returns:
        np.ndarray: Stacked 3D data.

    """
    paths = natsorted(glob.glob(os.path.join(pth, "*")))
    stacked = []
    for path in paths:
        img = Image.open(path)
        stacked.append(img)
    stacked = np.stack(stacked, axis=0)
    if format == "xyz":
        stacked = np.transpose(stacked, (1, 2, 0)).astype(np.int8)  # in (x, y, z) format

    return stacked


def save_slice_by_slice(pth: str, volume: np.ndarray, dim: int, format=".png"):
    """
    Save instance segmentation results.

    Args:
        pth: Path to save slices.
        volume: 3D data.
        dim: Along which dimension to save.
        format: Format of saved slice.

    """
    if volume.ndim != 3:
        raise ValueError("Volume should be 3D array")

    sample_name = pth.split('/')[-1]
    if not os.path.exists(pth):
        os.makedirs(pth)

    # Save slice by slice (0.png, 1.png, ...)
    for i in range(volume.shape[dim]):
        mask_img = Image.fromarray(volume[:, :, i].astype(np.uint8))
        mask_img.save(os.path.join(pth, str(i) + format))
