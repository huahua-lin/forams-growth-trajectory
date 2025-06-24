import argparse
import os

import torch.optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from datasets import ForamDataset3D

from utils import seed_worker, control_random
from unet3d import UNet3D
import torchio as tio


def run(args):
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    train_transform = tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2), p=0.5),  # Spatial flips
    ])

    train_dataset = ForamDataset3D(args.data_folder, phase="train", transform=train_transform, seg="unet")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               worker_init_fn=seed_worker,
                                               generator=torch.Generator().manual_seed(0), )

    model = UNet3D(in_channels=1, num_classes=1).to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(args.epochs):
        model.train()
        training_loss = 0.0
        for _, (x, y_true, _) in enumerate(train_loader):
            x, y_true = x.to(device=device, dtype=torch.float), y_true.to(device=device, dtype=torch.float)
            optimizer.zero_grad()
            y_pred = model(x)

            loss = bce_loss(y_pred, y_true)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()

        writer.add_scalar("BCE/Train", training_loss / len(train_loader), epoch)
        writer.flush()
        print("Epoch %d, average training BCE %9.6f" % (epoch + 1, training_loss / len(train_loader)))
        if args.save:
            os.makedirs(args.cpt, exist_ok=True)
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, args.cpt + "unet3d-{}.pth".format(epoch + 1))
    writer.close()


if __name__ == '__main__':
    control_random()

    parser = argparse.ArgumentParser(description="Training 3D U-Net model for semantic segmentation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data-folder", type=str, default=".", help="Folder containing training data")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--cpt", type=str, default="../../out/cpt/")
    parser.add_argument("--log-dir", type=str, default="../../out/train_and_test/")
    args = parser.parse_args()
    run(args)
