
import torch
import torch.nn as nn

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the Euclidean distance between the two outputs
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)

        # Calculate the contrastive loss
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss