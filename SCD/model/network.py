import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .backbone.mobilenetv2 import mobilenet_v2
from .backbone.resnet import resnet18, resnet50
from .block.bifpn import BiFPN
from .block.bifpn_add import BiFPN_add
from .block.neighbor import NeighborFeatureAggregation
from .block.fpn import FPN
from .block.fpn_plain import FPN_plain
from .block.vertical import VerticalFusion
from .block.convs import ConvBnRelu, DsBnRelu
from .util import init_method
from .block.heads import FCNHead, GatedResidualUpHead


def get_backbone(backbone_name):
    if backbone_name == 'mobilenetv2':
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
    elif backbone_name == 'mobilenetv3_small_075':
        backbone = timm.create_model('mobilenetv3_small_075', pretrained=True, features_only=True)
        backbone.channels = [16, 16, 24, 40, 432]
    elif backbone_name == 'mobilenetv3_small_100':
        backbone = timm.create_model('mobilenetv3_small_100', pretrained=True, features_only=True)
        backbone.channels = [16, 16, 24, 48, 576]
    elif backbone_name == 'resnet18':
        backbone = resnet18(pretrained=True, progress=True, replace_stride_with_dilation=[False, False, False])
        backbone.channels = [64, 64, 128, 256, 512]
    elif backbone_name == 'resnet18d':
        backbone = timm.create_model('resnet18d', pretrained=True, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
    elif backbone_name == 'resnet50':
        backbone = resnet50(pretrained=True, progress=True)
        backbone.channels = [64, 256, 512, 1024, 2048]
    elif backbone_name == 'hrnet_w18':
        backbone = timm.create_model('hrnet_w18', pretrained=True, features_only=True)
        backbone.channels = [64, 128, 256, 512, 1024]
    else:
        raise NotImplementedError("BACKBONE [%s] is not implemented!\n" % backbone_name)
    return backbone


def get_fpn(fpn_name, in_channels, out_channels, deform_groups=4, gamma_mode='SE', beta_mode='gatedconv'):
    if fpn_name == 'fpn':
        fpn = FPN(in_channels, out_channels, deform_groups, gamma_mode, beta_mode)
    elif fpn_name == 'fpn_plain':
        fpn = FPN_plain(in_channels, out_channels)
    elif fpn_name == 'bifpn':
        fpn = BiFPN(in_channels, out_channels)
    elif fpn_name == 'bifpn_add':
        fpn = BiFPN_add(in_channels, out_channels)
    elif fpn_name == 'neighbor':
        fpn = NeighborFeatureAggregation(in_channels, out_channels)
    else:
        raise NotImplementedError("FPN [%s] is not implemented!\n" % fpn_name)
    return fpn


class Detector(nn.Module):
    def __init__(self, backbone_name='mobilenetv2', fpn_name='fpn', fpn_channels=64,
                 deform_groups=4, gamma_mode='SE', beta_mode='contextgatedconv',
                 num_heads=1, num_points=8, kernel_layers=1, dropout_rate=0.1, init_type='kaiming_normal'):
        super().__init__()
        self.backbone = get_backbone(backbone_name)
        self.fpn = get_fpn(fpn_name, in_channels=self.backbone.channels[-4:], out_channels=fpn_channels,
                           deform_groups=deform_groups, gamma_mode=gamma_mode, beta_mode=beta_mode)
        self.p5_to_p4 = VerticalFusion(fpn_channels, num_heads=num_heads, num_points=4,
                                                    kernel_layers=kernel_layers)
        self.p4_to_p3 = VerticalFusion(fpn_channels, num_heads=num_heads, num_points=8,
                                                    kernel_layers=kernel_layers)
        self.p3_to_p2 = VerticalFusion(fpn_channels, num_heads=num_heads, num_points=16,
                                                    kernel_layers=kernel_layers)

        self.p5_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p4_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p3_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p2_head = nn.Conv2d(fpn_channels, 2, 1)
        self.project = nn.Sequential(nn.Conv2d(fpn_channels * 4, fpn_channels, 1, bias=False),
                                     nn.BatchNorm2d(fpn_channels),
                                     nn.ReLU(True)
                                     )
        self.head = GatedResidualUpHead(fpn_channels, 2, dropout_rate=dropout_rate)
        self.scd_head = FCNHead(fpn_channels, 7, dropout_rate=dropout_rate)
        # init_method(self.fpn, self.p5_to_p4, self.p4_to_p3, self.p3_to_p2, self.p5_head, self.p4_head,
        #             self.p3_head, self.p2_head, init_type=init_type)

    def forward(self, x1, x2):
        ### Extract backbone features
        t1_c1, t1_c2, t1_c3, t1_c4, t1_c5 = self.backbone.forward(x1)
        t2_c1, t2_c2, t2_c3, t2_c4, t2_c5 = self.backbone.forward(x2)
        t1_p2, t1_p3, t1_p4, t1_p5 = self.fpn([t1_c2, t1_c3, t1_c4, t1_c5])
        t2_p2, t2_p3, t2_p4, t2_p5 = self.fpn([t2_c2, t2_c3, t2_c4, t2_c5])

        diff_p2 = torch.abs(t1_p2 - t2_p2)
        diff_p3 = torch.abs(t1_p3 - t2_p3)
        diff_p4 = torch.abs(t1_p4 - t2_p4)
        diff_p5 = torch.abs(t1_p5 - t2_p5)
        """
        pred_p5 = self.p5_head(diff_p5)
        pred_p4 = self.p4_head(diff_p4)
        pred_p3 = self.p3_head(diff_p3)
        pred_p2 = self.p2_head(diff_p2)

        diff_p3 = F.interpolate(diff_p3, size=(64, 64), mode='bilinear', align_corners=False)
        diff_p4 = F.interpolate(diff_p4, size=(64, 64), mode='bilinear', align_corners=False)
        diff_p5 = F.interpolate(diff_p5, size=(64, 64), mode='bilinear', align_corners=False)
        #diff = diff_p2 + diff_p3 + diff_p4 + diff_p5
        diff = torch.cat([diff_p2, diff_p3, diff_p4, diff_p5], dim=1)
        diff = self.project(diff)
        pred = self.head(diff)

        """
        fea_p5 = diff_p5
        pred_p5 = self.p5_head(fea_p5)
        fea_p4 = self.p5_to_p4(fea_p5, diff_p4)
        pred_p4 = self.p4_head(fea_p4)
        fea_p3 = self.p4_to_p3(fea_p4, diff_p3)
        pred_p3 = self.p3_head(fea_p3)
        fea_p2 = self.p3_to_p2(fea_p3, diff_p2)
        pred_p2 = self.p2_head(fea_p2)
        #fea_p3 = F.interpolate(fea_p3, size=(64, 64), mode='bilinear', align_corners=False)
        #fea_p4 = F.interpolate(fea_p4, size=(64, 64), mode='bilinear', align_corners=False)
        #fea_p5 = F.interpolate(fea_p5, size=(64, 64), mode='bilinear', align_corners=False)
        #diff = diff_p2 + diff_p3 + diff_p4 + diff_p5
        #diff = torch.cat([fea_p2, fea_p3, fea_p4, fea_p5], dim=1)
        #diff = self.project(diff)
        pred = self.head(fea_p2)


        pred_p2 = F.interpolate(pred_p2, size=(256, 256), mode='bilinear', align_corners=False)
        pred_p3 = F.interpolate(pred_p3, size=(256, 256), mode='bilinear', align_corners=False)
        pred_p4 = F.interpolate(pred_p4, size=(256, 256), mode='bilinear', align_corners=False)
        pred_p5 = F.interpolate(pred_p5, size=(256, 256), mode='bilinear', align_corners=False)
        #pred = F.interpolate(pred, size=(256, 256), mode='bilinear', align_corners=False)

        t1_p3 = F.interpolate(t1_p3, size=(64, 64), mode='bilinear', align_corners=False)
        t1_p4 = F.interpolate(t1_p4, size=(64, 64), mode='bilinear', align_corners=False)
        t1_p5 = F.interpolate(t1_p5, size=(64, 64), mode='bilinear', align_corners=False)
        t1_fea = torch.cat([t1_p2, t1_p3, t1_p4, t1_p5], dim=1)
        t1_fea = self.project(t1_fea)
        pred_seg1 = self.scd_head(t1_fea)
        pred_seg1 = F.interpolate(pred_seg1, size=(256, 256), mode='bilinear', align_corners=False)
        t2_p3 = F.interpolate(t2_p3, size=(64, 64), mode='bilinear', align_corners=False)
        t2_p4 = F.interpolate(t2_p4, size=(64, 64), mode='bilinear', align_corners=False)
        t2_p5 = F.interpolate(t2_p5, size=(64, 64), mode='bilinear', align_corners=False)
        t2_fea = torch.cat([t2_p2, t2_p3, t2_p4, t2_p5], dim=1)
        t2_fea = self.project(t2_fea)
        pred_seg2 = self.scd_head(t2_fea)
        pred_seg2 = F.interpolate(pred_seg2, size=(256, 256), mode='bilinear', align_corners=False)

        return pred, pred_seg1, pred_seg2, pred_p2, pred_p3, pred_p4, pred_p5



