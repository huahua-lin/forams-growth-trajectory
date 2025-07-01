import argparse
import os

import numpy as np
from skimage.metrics import adapted_rand_error, variation_of_information
from utils import stack_imgs, resize_volume_with_aspect_ratio


def run():
    names = os.listdir(args.gt_seg_pth)
    error_list = []
    splits_list = []
    merges_list = []
    jaccard_list = []
    for name in names:
        print(name)
        pred_stack = stack_imgs(os.path.join(args.pred_seg_pth, name))
        gt_stack = stack_imgs(os.path.join(args.gt_seg_pth, name))
        gt_stack = resize_volume_with_aspect_ratio(gt_stack, (128, 128, 64))  # resize if needed

        mask = (gt_stack != 0) & (pred_stack != 0)

        error, _, _ = adapted_rand_error(gt_stack[mask], pred_stack[mask], ignore_labels=0)
        error_list.append(error)
        print("Adapted Rand Error: {:.3f}".format(error))

        splits, merges = variation_of_information(gt_stack[mask], pred_stack[mask], ignore_labels=0)
        splits_list.append(splits)
        merges_list.append(merges)
        print("False Splits: {:.3f}".format(splits))
        print("False Merges: {:.3f}".format(merges))

        intersection = np.logical_and(gt_stack, pred_stack).sum()
        union = np.logical_or(gt_stack, pred_stack).sum()
        jaccard = intersection / union
        jaccard_list.append(jaccard)
        print("Jaccard Score: {:.3f}".format(jaccard))

    print(error_list)
    print(splits_list)
    print(merges_list)
    print(jaccard_list)
    print("Average Rand Error:", sum(error_list) / len(error_list))
    print("Average Rand Splits:", sum(splits_list) / len(splits_list))
    print("Average Rand Merges:", sum(merges_list) / len(merges_list))
    print("Average Jaccard Score:", sum(jaccard_list) / len(jaccard_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluating the prediction of instance segmentation")
    parser.add_argument("--pred-seg-pth", type=str,
                        default=".",
                        help="A directory that stores the predicted instance segmentation results divided by samples")
    parser.add_argument("--gt-seg-pth", type=str,
                        default=".",
                        help="A directory that stores the ground truth instance segmentation results divided by samples")
    args = parser.parse_args()
    run()
