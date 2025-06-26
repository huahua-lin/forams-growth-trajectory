import argparse
import os
from pathlib import Path

import torch

from unet3d import UNet3D
from datasets import ForamDataset3D
from utils import save_slice_by_slice


def run():
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    test_dataset = ForamDataset3D(args.data_pth, phase="test", transform=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet3D(in_channels=1, num_classes=1).to(device)

    checkpoint = torch.load(args.cpt, map_location=device, weights_only=True)
    for key in list(checkpoint['model_state_dict'].keys()):
        if 'module.' in key:
            checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'][key]
            del checkpoint['model_state_dict'][key]
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    for step, (x, _, filenames) in enumerate(test_loader):
        print(step, filenames)
        with torch.no_grad():
            y_pred = model(x.to(dtype=torch.float32, device=device))
            mask = torch.sigmoid(y_pred) > 0.5
            mask = mask.squeeze(1)
            print(torch.unique(mask))

        # save the predicted mask as a PNG image
        if args.save:
            for i in range(len(filenames)):  # loop batch
                Path(os.path.join(args.save_loc, filenames[i])).mkdir(parents=True, exist_ok=True)
                save_slice_by_slice(os.path.join(args.save_loc, filenames[i]), mask[i].cpu().numpy(), dim=0,
                                    format=".png")
            print("Images saved.")


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
