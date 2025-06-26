import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary logits input.

        Args:
            alpha: balancing factor for class imbalance
            gamma: focusing parameter
            reduction: 'none' | 'mean' | 'sum'

        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: raw outputs from the network, shape [B, 1, ...]
        targets: binary ground truth, same shape
        """
        probs = torch.sigmoid(logits)
        targets = targets.type_as(logits)

        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # combines sigmoid + BCE
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_term = (1 - pt) ** self.gamma

        loss = self.alpha * focal_term * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        """
        Computes the Dice Loss between the predicted and ground truth tensors.

        Args:
            y_pred (torch.Tensor): The predicted output from the model.
                                  Expected shape: (batch_size, 1, ...)
            y_true (torch.Tensor): The ground truth labels.
                                  Expected shape: (batch_size, 1, ...)

        Returns:
            torch.Tensor: A scalar loss value representing 1 - Dice coefficient.

       """
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
                y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc
