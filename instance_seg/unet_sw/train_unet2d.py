import argparse
import os

import torch.optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from datasets import ForamDataset2D

from utils import seed_worker, control_random
from unet2d import UNet2D


def run():
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    train_dataset = ForamDataset2D(args.data_pth, phase="train", transform=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               worker_init_fn=seed_worker,
                                               generator=torch.Generator().manual_seed(0), )

    model = UNet2D(n_channels=1, n_classes=1).to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
                        }, args.cpt + "unet2d-{}.pth".format(epoch + 1))
    writer.close()


if __name__ == '__main__':
    control_random()

    parser = argparse.ArgumentParser(description="Training 2D U-Net model for semantic segmentation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data-pth", type=str, default=".", help="Path to training data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--cpt", type=str, default="../../out/cpt/")
    parser.add_argument("--log-dir", type=str, default="../../out/train_and_test/")
    args = parser.parse_args()
    run()
