import argparse
import os
from pathlib import Path

import torch

from unet3d_mtl import UNet3D_MTL
from datasets import ForamDataset3D
from utils import save_slice_by_slice


def run():
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    test_dataset = ForamDataset3D(args.data_folder, phase="test", transform=None, seg="mtl")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet3D_MTL(in_channels=1, num_classes=1).to(device)

    checkpoint = torch.load(args.cpt, map_location=device, weights_only=True)
    for key in list(checkpoint['model_state_dict'].keys()):
        if 'module.' in key:
            checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'][key]
            del checkpoint['model_state_dict'][key]
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    for step, (x, _, filenames) in enumerate(test_loader):
        print(step, filenames)
        x = x.to(dtype=torch.float32, device=device)

        # make predictions
        with torch.no_grad():
            fg, bdry, bg = model(x)

        # generate binary masks
        fg_pred = torch.sigmoid(fg) > 0.5
        bdry_pred = torch.sigmoid(bdry) > 0.4
        bg_pred = torch.sigmoid(bg) > 0.9
        mask = (fg_pred & (~bdry_pred)) & (~bg_pred)
        mask = mask.squeeze(1)

        # save the masks as PNG images
        if args.save:
            for i in range(len(filenames)):  # loop batch
                Path(os.path.join(args.save_loc, filenames[i])).mkdir(parents=True, exist_ok=True)
                save_slice_by_slice(os.path.join(args.save_loc, filenames[i]), mask[i].cpu().numpy(), dim=0,
                                    format=".png")
            print("Images saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training model for segmentation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data-folder", type=str, default=".", help="Path to testing data")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpt", type=str, default=".", help="Path to checkpoint")
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--save-loc", type=str, default=".", help="Path to save images")
    args = parser.parse_args()
    run()
