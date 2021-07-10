import torch.nn as nn
import torch.nn.functional as F
import torch


# Make custom focal loss.
class MyFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(MyFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_Loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="sum")
        targets = targets.type(torch.long)
        # at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_Loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_Loss
        return F_loss.sum()