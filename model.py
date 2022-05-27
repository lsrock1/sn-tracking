import torchvision

from torch import nn
import torch
import torch.nn.functional as F

from collections import OrderedDict
import copy


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


class Model2(nn.Module):
    def __init__(self, ):
        super().__init__()
        model = get_model()
        self.backbone = model.backbone
        self.classifier = model.classifier
        # copy aspp
        self.re_id = copy.deepcopy(self.classifier[0])
        self.re_id.project[0] = nn.Conv2d(1280, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.re_id.project[1] = nn.BatchNorm2d(1024, 1024)
        self.re_id.project[3] = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=True)
        self.weight = nn.Parameter(torch.randn(1, 1, 1024, 1, 1))

    def reid_run(self, x, kernel):
        bs, c, h, w = x.shape
        # x = F.interpolate(x, size=(32, 57), mode="bilinear", align_corners=False)
        # bs, n+1, c
        background = self.weight.expand(kernel.shape[0], 1, -1, -1, -1)
        kernel = torch.cat([background, kernel], dim=1).reshape(bs, -1, c)
        # print(x.reshape(bs, c, -1).transpose(1, 2)[..., None].shape)
        # bs, hw, c * bs, c, n+!
        # seg_map = F.cosine_similarity(
        #     x.reshape(bs, c, -1).transpose(1, 2)[..., None],
        #     kernel.transpose(1, 2).unsqueeze(1), dim=2
        # )
        seg_map = torch.bmm(
            x.reshape(bs, c, -1).transpose(1, 2),
            kernel.transpose(1, 2)
        )
        seg_map = seg_map.transpose(1, 2).reshape(bs, -1, h, w)
        # seg_map = F.interpolate(seg_map, size=(h, w), mode="bilinear", align_corners=False)
        # batch, n, channel = kernel.shape[:3]
        # # x = x.transpose(0, 1)
        # x = x.reshape(1, batch*channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
        # kernel = kernel.reshape(batch*n, channel, 1, 1)  # (b*c) * 1 * H * W
        # out = F.conv2d(x, kernel, groups=batch)
        # out = out.reshape(batch, n, out.size(2), out.size(3))
        return seg_map

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        x = self.re_id(features["out"])
        # print(x.shape)
        
        # x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result['feat'] = x
        return result


def get_modelv2():
    m = Model2()
    return m


def get_model():
    m = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    m.aux_classifier = None
    m.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    m.backbone.conv1 = Stem(m.backbone.conv1, m.backbone.bn1)
    m.backbone.bn1 = nn.Identity()
    return m
