import torch
import torch.nn as nn
import torch.nn.functional as F
from resblock import LazyBottleneck, lazyconv1x1


class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()
        self.bottleneck = nn.ModuleList(
            [LazyBottleneck(32, downsample=lazyconv1x1(32)) for _ in range(5)]
        )
        self.conv_out = nn.Conv2d(5 * 32, 1, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_hat, features, y):
        masks = []
        for feat, bottleneck in zip(features, self.bottleneck):
            _ = bottleneck(feat)
            _ = F.interpolate(_, (256, 256), mode="bilinear", align_corners=True)
            masks.append(_)

        masks = torch.cat(masks, dim=1)
        masks = self.conv_out(masks)
        masks = self.sigmoid(masks)

        return masks * y_hat + (1 - masks) * y, masks
