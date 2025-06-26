from typing import Optional

import numpy as np
from elf.segmentation.watershed import apply_size_filter
from elf.segmentation import compute_boundary_mean_and_length
from elf.segmentation.multicut import transform_probabilities_to_costs
from elf.segmentation import GaspFromAffinities


def shift_affinities(affinities, offsets):
    rolled_affs = []
    for i, _ in enumerate(offsets):
        offset = offsets[i]
        shifts = tuple([int(off / 2) for off in offset])

        padding = [[0, 0] for _ in range(len(shifts))]
        for ax, shf in enumerate(shifts):
            if shf < 0:
                padding[ax][1] = -shf
            elif shf > 0:
                padding[ax][0] = shf

        padded_inverted_affs = np.pad(affinities, pad_width=((0, 0),) + tuple(padding), mode='constant')

        crop_slices = tuple(
            slice(padding[ax][0], padded_inverted_affs.shape[ax + 1] - padding[ax][1]) for ax in range(3)
        )

        padded_inverted_affs = np.roll(padded_inverted_affs[i], shifts, axis=(0, 1, 2))[crop_slices]
        rolled_affs.append(padded_inverted_affs)
        del padded_inverted_affs

    rolled_affs = np.stack(rolled_affs)
    return rolled_affs


def gasp(
        boundary_pmaps: np.ndarray,
        superpixels: Optional[np.ndarray] = None,
        gasp_linkage_criteria: str = 'average',  # mutex_watershed or average
        beta: float = 0.9,  # higher beta tends to over-segment
        post_minsize: int = 10,
        n_threads: int = 6,
) -> np.ndarray:
    """
    Perform segmentation using the GASP algorithm with affinity maps.

    Args:
        boundary_pmaps (np.ndarray): Cell boundary prediction.
        superpixels (Optional[np.ndarray]): Superpixel segmentation. If None, GASP will be run from the pixels. Default is None.
        gasp_linkage_criteria (str): Linkage criteria for GASP. Default is 'average'.
        beta (float): Beta parameter for GASP. Small values steer towards under-segmentation, while high values bias towards over-segmentation. Default is 0.5.
        post_minsize (int): Minimum size of the segments after GASP. Default is 100.
        n_threads (int): Number of threads used for GASP. Default is 6.

    Returns:
        np.ndarray: GASP output segmentation.
    """
    remove_singleton = False
    if superpixels is not None:
        assert boundary_pmaps.shape == superpixels.shape, "Shape mismatch between boundary_pmaps and superpixels."
        if superpixels.ndim == 2:  # Ensure superpixels is 3D if provided
            superpixels = superpixels[None, ...]
            boundary_pmaps = boundary_pmaps[None, ...]
            remove_singleton = True

    # Prepare the arguments for running GASP
    run_GASP_kwargs = {
        'linkage_criteria': gasp_linkage_criteria,
        'add_cannot_link_constraints': False,
        'use_efficient_implementations': False,
    }

    # Interpret boundary_pmaps as affinities and prepare for GASP
    boundary_pmaps = boundary_pmaps.astype('float32')
    affinities = np.stack([boundary_pmaps] * 3, axis=0)

    offsets = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    # Shift is required to correct aligned affinities
    affinities = shift_affinities(affinities, offsets=offsets)

    # invert affinities
    affinities = 1 - affinities

    # Initialize and run GASP
    gasp_instance = GaspFromAffinities(
        offsets,
        superpixel_generator=None if superpixels is None else (lambda *args, **kwargs: superpixels),
        run_GASP_kwargs=run_GASP_kwargs,
        n_threads=n_threads,
        beta_bias=beta,
    )
    segmentation, _ = gasp_instance(affinities)

    # Apply size filtering if specified
    if post_minsize > 0:
        segmentation, _ = apply_size_filter(segmentation.astype('uint32'), boundary_pmaps, post_minsize)

    if remove_singleton:
        segmentation = segmentation[0]

    return segmentation