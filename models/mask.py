import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resblock import LazyBottleneck, lazyconv1x1


class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()
        self.bottleneck = nn.ModuleList(
            [
                LazyBottleneck(
                    32, downsample=lazyconv1x1(32), norm_layer=nn.InstanceNorm2d
                )
                for _ in range(5)
            ]
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

        fixed_from_generated = 0.5  # TODO: design a scheduler for this
        rest = 1 - fixed_from_generated

        fixed_part = (
            torch.empty_like(masks).fill_(fixed_from_generated).requires_grad_(True)
        )
        rest_mask = torch.empty_like(masks).fill_(rest).requires_grad_(True)

        fixed = fixed_part * y_hat
        rest = (rest_mask * masks) * y_hat + (rest_mask * (1 - y_hat)) * y

        return fixed + rest, masks
