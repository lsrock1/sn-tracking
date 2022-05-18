import torchvision

from torch import nn
import torch


class Stem(nn.Module):
    def __init__(self, conv, bn,):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.compressor = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, 32))
        self.compressor2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, 32))

    def forward(self, x):
        origin_bs = x.shape[0]
        x = torch.cat([x[:, :3], x[:, 3:]], dim=0)
        x = self.bn(self.conv(x))
        x = torch.cat([
            self.compressor(x[:origin_bs] + x[origin_bs:]),
            self.compressor2(x[origin_bs:] - x[:origin_bs])
        ], dim=1)
        return x


def get_model():
    m = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    m.aux_classifier = None
    m.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    m.backbone.conv1 = Stem(m.backbone.conv1, m.backbone.bn1)
    m.backbone.bn1 = nn.Identity()
    return m
