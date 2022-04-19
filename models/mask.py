import torch
import torch.nn as nn
import torch.nn.functional as F


class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()
        self.bottleneck = nn.ModuleList([nn.LazyConv2d(32, 1, 1, 0) for _ in range(5)])
        self.convs = nn.ModuleList([nn.LazyConv2d(32, 3, 1, 1) for _ in range(5)])
        self.conv_out = nn.Conv2d(5 * 32, 1, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_hat, features, y):
        masks = []
        for feat, conv, bottleneck in zip(features, self.convs, self.bottleneck):
            _ = conv(feat)
            _ += bottleneck(feat)
            _ = F.interpolate(_, (256, 256), mode="bilinear", align_corners=True)
            masks.append(_)

        masks = torch.cat(masks, dim=1)
        masks = self.conv_out(masks)
        masks = self.sigmoid(masks)

        return masks * y_hat + (1 - masks) * y, masks
