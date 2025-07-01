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
        img = np.array(Image.open(path))
        stacked.append(img)
    stacked = np.stack(stacked, axis=0)
    if format == "xyz":
        stacked = np.transpose(stacked, (1, 2, 0))  # in (x, y, z) format

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

    if not os.path.exists(pth):
        os.makedirs(pth)

    # Save slice by slice (0.png, 1.png, ...)
    for i in range(volume.shape[dim]):
        if dim == 0:
            slice_ = volume[i, :, :]
        elif dim == 1:
            slice_ = volume[:, i, :]
        elif dim == 2:
            slice_ = volume[:, :, i]
        else:
            raise ValueError("Invalid dim value; should be 0, 1, or 2")
        if np.issubdtype(volume.dtype, np.bool_):
            slice_ = Image.fromarray(slice_.astype(np.uint8) * 255).convert("1")  # save binary seg results: True, False
        elif np.issubdtype(slice_.dtype, np.floating):
            slice_ = Image.fromarray((slice_ * 255).astype(np.uint8))  # save probability maps: \in[0, 1]
        else:
            slice_ = Image.fromarray(slice_.astype(np.uint8))  # save instance seg results: 0, 1, 2, ...
        slice_.save(os.path.join(pth, str(i) + format))
