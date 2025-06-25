import argparse
import os

import torch
from PIL import Image

from unet3d import UNet3D
from utils import *
from datasets import ForamDataset3D


def predict_mask(net, img, threshold=0.5):
    net.eval()
    with torch.no_grad():
        y_pred = net(img)
        mask = torch.sigmoid(y_pred) > threshold
    return mask.squeeze(1)


def run():
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    test_dataset = ForamDataset3D(args.data_folder, phase="test", transform=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    net = UNet3D(in_channels=1, num_classes=1).to(device)

    checkpoint = torch.load(args.cpt, map_location=device, weights_only=True)
    for key in list(checkpoint['model_state_dict'].keys()):
        if 'module.' in key:
            checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'][key]
            del checkpoint['model_state_dict'][key]
    net.load_state_dict(checkpoint['model_state_dict'])

    for step, (x, _, filenames) in enumerate(test_loader):
        print(step, filenames)
        mask = predict_mask(net, x.to(dtype=torch.float, device=device))

        # save the predicted mask as a PNG image
        if args.save:
            if not os.path.exists(args.save_loc):
                os.makedirs(args.save_loc)
            for i in range(len(filenames)):
                if not os.path.exists(os.path.join(args.save_loc, filenames[i])):
                    os.makedirs(os.path.join(args.save_loc, filenames[i]))
                for j in range(mask.shape[1]):
                    mask_img = Image.fromarray(mask[i, j, :, :].cpu().numpy()).convert("1")
                    mask_img.save(os.path.join(args.save_loc, os.path.join(filenames[i], str(j) + ".png")))


if __name__ == '__main__':
    control_random()

    parser = argparse.ArgumentParser(description="Predicting semantic segmentation results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data-folder", type=str, default=".", help="Path to testing data")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpt", type=str, default=".", help="Path to checkpoint")
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--save-loc", type=str, default=".", help="Path to save results")
    args = parser.parse_args()
    run()
