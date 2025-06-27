import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from unet2d import UNet2D
from datasets import ForamDataset2D


def run():
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    test_dataset = ForamDataset2D(args.data_pth, phase="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet2D(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load(args.cpt, map_location=device, weights_only=True))
    model.eval()

    for step, (x, _, filenames) in enumerate(test_loader):
        print(step, filenames)
        x = x.to(dtype=torch.float32, device=device)

        # make predictions
        with torch.no_grad():
            y_pred = model(x)

        # generate binary masks
        mask = torch.sigmoid(y_pred) > 0.5

        # save the masks as PNG images
        if args.save:
            for i in range(len(filenames)):  # loop batch
                parts = os.path.normpath(filenames[i]).split(os.sep)
                Path(os.path.join(args.save_loc, parts[0])).mkdir(parents=True, exist_ok=True)
                # convert Bool array to 1-bit images
                mask_img = Image.fromarray(mask[i].squeeze().cpu().numpy().astype(np.uint8) * 255).convert("1")
                mask_img.save(os.path.join(args.save_loc, filenames[i]))
            print("Image saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predicting semantic segmentation results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data-pth", type=str, default=".", help="Path to testing data")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpt", type=str, default=".", help="Path to checkpoint")
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--save-loc", type=str, default=".", help="Path to save images")

    args = parser.parse_args()
    run()
