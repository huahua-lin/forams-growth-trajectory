import argparse
import os

import numpy as np
import torch
from PIL import Image

from unet2d import UNet2D
from utils import seed_worker, control_random
from datasets import ForamDataset2D


def predict_mask(net, img, threshold=0.5):
    net.eval()
    with torch.no_grad():
        y_pred = net(img)
        mask = torch.sigmoid(y_pred) > threshold
    return mask


def mask_to_img(mask):
    out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.float32)
    out[np.where(mask)] = 255
    return Image.fromarray(out).convert("1")


def run():
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    test_dataset = ForamDataset2D(args.data_pth, phase="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(0), )

    model = UNet2D(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load(args.cpt, map_location=device, weights_only=True))

    for step, (x, _, filenames) in enumerate(test_loader):
        print(step, filenames)
        mask = predict_mask(model, x.to(dtype=torch.float, device=device))

        # save the predicted mask as a PNG image
        if args.save:
            if not os.path.exists(args.save_loc):
                os.makedirs(args.save_loc)
            for i in range(len(filenames)):
                mask_img = mask_to_img(mask[i].squeeze().cpu().numpy())
                foram_ID = filenames[i].split('/')[0]
                if not os.path.exists(os.path.join(args.save_loc, foram_ID)):
                    os.makedirs(os.path.join(args.save_loc, foram_ID))
                mask_img.save(os.path.join(args.save_loc, filenames[i]))


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
