# ...existing code...
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Union, Dict

class ImageEncoder(torch.nn.Module):
    def __init__(self, config: Union[int, Dict]=512, pretrained: bool=True):
        """
        Accept either:
         - output_dim (int)
         - config dict with keys: output_size or output_dim, pretrained (optional)
        """
        super(ImageEncoder, self).__init__()
        if isinstance(config, dict):
            output_dim = int(config.get("output_size", config.get("output_dim", 512)))
            pretrained = bool(config.get("pretrained", pretrained))
        else:
            output_dim = int(config)

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None) if hasattr(models, "ResNet18_Weights") else models.resnet18(pretrained=pretrained)
        in_features = resnet.fc.in_features
        # remove final fc, keep feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # outputs [B, in_features, 1, 1]
        self.fc = nn.Linear(in_features, output_dim)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        return out

    def encode(self, images):
        return self.forward(images)

    def extract_features(self, images):
        feat = self.backbone(images)
        return feat.view(feat.size(0), -1)
# ...existing code...