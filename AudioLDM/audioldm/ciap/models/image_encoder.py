import torch
import torch.nn as nn
import torchvision.models as models
from typing import Union, Dict

class ImageEncoder(nn.Module):
    """
    ResNet18 backbone + projection head to produce CLAP-style 512-d embeddings.
    Accept config as int (output_dim) or dict with keys: output_dim, pretrained (bool)
    """
    def __init__(self, config: Union[int, Dict]=512, pretrained: bool=True):
        super(ImageEncoder, self).__init__()
        # parse config
        if isinstance(config, dict):
            output_dim = int(config.get("output_dim", config.get("output_size", 512)))
            pretrained = bool(config.get("pretrained", pretrained))
        else:
            output_dim = int(config)

        # backbone
        if hasattr(models, "ResNet18_Weights"):
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            resnet = models.resnet18(pretrained=pretrained)

        in_features = resnet.fc.in_features
        # remove final fc, keep feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # outputs [B, in_features, 1, 1]
        self.output_dim = output_dim
        # projection head to desired embedding dim
        self.proj = nn.Linear(in_features, output_dim)
        # optional layernorm
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        feats = self.extract_features(x)          # [B, in_features]
        out = self.proj(feats)                    # [B, output_dim]
        out = self.ln(out)
        return out

    def extract_features(self, images):
        # returns flattened backbone features [B, in_features]
        x = images
        # pass through backbone
        z = self.backbone(x)                      # [B, in_features, 1, 1]
        z = z.view(z.size(0), -1)
        return z

    def encode(self, images):
        """
        CLAP-style API: returns [B, D] embeddings (not normalized here).
        Normalization / L2 can be applied by wrapper.
        """
        with torch.no_grad():
            out = self.forward(images)
        return out