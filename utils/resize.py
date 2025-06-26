from typing import Tuple

from skimage.transform import resize
import numpy as np


def resize_volume_with_aspect_ratio(volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
    """
    Resize a 3D volume to fit within a target size while preserving its aspect ratio,
    then pad it to exactly match the target size.

    Args:
        volume: The input 3D volume with shape (D, H, W).
        target_size: The desired output size as (D, H, W).

    Returns:
        np.ndarray: The resized and padded volume with shape equal to target_size.

    """
    scale_depth = target_size[0] / volume.shape[0]
    scale_height = target_size[1] / volume.shape[1]
    scale_width = target_size[2] / volume.shape[2]

    # Use the smallest scale factor to preserve the aspect ratio
    scale = min(scale_depth, scale_height, scale_width)
    new_depth = int(volume.shape[0] * scale)
    new_height = int(volume.shape[1] * scale)
    new_width = int(volume.shape[2] * scale)

    # Resize the volume
    resized_volume = resize(
        volume,
        (new_depth, new_height, new_width),
        mode='constant',
        preserve_range=True,
        order=0
    )

    # Pad the volume to match the target size
    pad_depth = (target_size[0] - new_depth) // 2
    pad_height = (target_size[1] - new_height) // 2
    pad_width = (target_size[2] - new_width) // 2

    padded_volume = np.pad(
        resized_volume,
        pad_width=((pad_depth, target_size[0] - new_depth - pad_depth),
                   (pad_height, target_size[1] - new_height - pad_height),
                   (pad_width, target_size[2] - new_width - pad_width)),
        mode='constant',
        constant_values=0
    )

    return padded_volume


def resize_img_with_aspect_ratio(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a 2D image to fit within a target size while preserving its aspect ratio,
    and pad the result with zeros to exactly match the target size.

    Args:
        img: The input 2D image (height x width).
        target_size The desired output size as (target_height, target_width).

    Returns:
        np.ndarray: The resized and zero-padded image with shape equal to target_size.

    """
    scale_height = target_size[0] / img.shape[0]
    scale_width = target_size[1] / img.shape[1]
    scale = min(scale_height, scale_width)

    scaled_img = resize(img, (int(img.shape[0] * scale), int(img.shape[1] * scale)), mode='constant',
                        preserve_range=True, order=0)
    output = np.zeros((target_size[0], target_size[1]), dtype=np.float32)
    scaled_h, scaled_w = scaled_img.shape
    output[:scaled_w, :scaled_h] = np.array(scaled_img)

    return output
