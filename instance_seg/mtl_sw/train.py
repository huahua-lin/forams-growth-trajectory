import argparse
import os

import torch.optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from datasets import ForamDataset3D

from utils import FocalLoss, seed_worker, control_random
from unet3d_mtl import UNet3D_MTL
import torchio as tio


def run():
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    g = torch.Generator()
    g.manual_seed(0)

    train_transform = tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2), p=0.5),  # Spatial flips
    ])

    train_dataset = ForamDataset3D(args.data_folder, phase="train", transform=train_transform, seg="mtl")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               worker_init_fn=seed_worker,
                                               generator=torch.Generator().manual_seed(0), )

    model = UNet3D_MTL(in_channels=1, num_classes=1).to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        training_loss = 0.0
        train_fg_loss = 0.0
        train_bg_loss = 0.0
        train_bdry_loss = 0.0
        train_consistence_loss = 0.0
        for _, (x, mask, _) in enumerate(train_loader):
            x, fg_mask, bdry_mask, bg_mask = x.to(device=device, dtype=torch.float), mask[0].to(device=device,
                                                                                                dtype=torch.float), mask[1].to(
                device=device, dtype=torch.float), mask[2].to(device=device, dtype=torch.float)
            optimizer.zero_grad()
            fg, bdry, bg = model(x)

            fg_loss = FocalLoss(alpha=1.0, gamma=2.0)(fg, fg_mask)
            bdry_loss = FocalLoss(alpha=1.0, gamma=4.0)(bdry, bdry_mask)
            bg_loss = FocalLoss(alpha=0.5, gamma=2.0)(bg, bg_mask)

            fg_mask = fg_mask.squeeze(1)
            bg_mask = bg_mask.squeeze(1)
            bdry_mask = bdry_mask.squeeze(1)

            logits = torch.cat([bg, fg, bdry], dim=1)
            targets = torch.zeros_like(bg_mask)
            targets[bdry_mask == 1] = 2
            targets[fg_mask == 1] = 1

            consistence_loss = ce_loss(logits, targets.long())

            loss = fg_loss + 2.0 * bdry_loss + 0.5 * bg_loss + consistence_loss

            training_loss += loss.item()
            train_fg_loss += fg_loss.item()
            train_bg_loss += bg_loss.item()
            train_bdry_loss += bdry_loss.item()
            train_consistence_loss += consistence_loss.item()
            loss.backward()
            optimizer.step()

        writer.add_scalar("Total_Loss", training_loss / len(train_loader), epoch)
        writer.add_scalar("FG_Loss", train_fg_loss / len(train_loader), epoch)
        writer.add_scalar("BG_Loss", train_bg_loss / len(train_loader), epoch)
        writer.add_scalar("BDRY_Loss", train_bdry_loss / len(train_loader), epoch)
        writer.add_scalar("Consistence_Loss", train_consistence_loss / len(train_loader), epoch)
        writer.flush()
        print("Epoch %d, average training loss %9.6f" % (epoch + 1, training_loss / len(train_loader)))
        torch.save({'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, args.cpt + "mtl3d-{}.pth".format(epoch + 1))
    writer.close()


if __name__ == '__main__':
    control_random()

    parser = argparse.ArgumentParser(description="Training 3D U-Net model for semantic segmentation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data-folder", type=str, default=".", help="Folder containing training data")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--cpt", type=str, default="../../out/cpt/")
    parser.add_argument("--log-dir", type=str, default="../../out/train_and_test/")
    args = parser.parse_args()
    run()
