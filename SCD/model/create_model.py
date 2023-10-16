from .network import Detector
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import os
import torch.optim as optim
from .block.schedular import get_cosine_schedule_with_warmup
from .loss.bcedice import BCEDiceLoss
from .loss.focal import BinaryFocalLoss, FocalLoss
from .loss.dice import BinaryDICELoss, DICELoss
from thop import profile


def get_model(backbone_name='mobilenetv2', fpn_name='neighbor', fpn_channels=64, deform_groups=4,
              gamma_mode='SE', beta_mode='gatedconv', num_heads=1, num_points=8, kernel_layers=1,
              dropout_rate=0.1, init_type='kaiming_normal'):
    detector = Detector(backbone_name, fpn_name, fpn_channels, deform_groups, gamma_mode, beta_mode,
                        num_heads, num_points, kernel_layers, dropout_rate, init_type)
    print(detector)
    input1 = torch.randn(1, 3, 256, 256)
    input2 = torch.randn(1, 3, 256, 256)
    flops, params = profile(detector, inputs=(input1, input2))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

    return detector


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.device = torch.device("cuda:%s" % opt.gpu_ids[0] if torch.cuda.is_available() else "cpu")
        self.opt = opt
        self.base_lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.detector = get_model(backbone_name=opt.backbone, fpn_name=opt.fpn, fpn_channels=opt.fpn_channels,
                                  deform_groups=opt.deform_groups, gamma_mode=opt.gamma_mode, beta_mode=opt.beta_mode,
                                  num_heads=opt.num_heads, num_points=opt.num_points, kernel_layers=opt.kernel_layers,
                                  dropout_rate=opt.dropout_rate, init_type=opt.init_type)
        self.cd_focal = BinaryFocalLoss(alpha=opt.alpha, gamma=opt.gamma)
        self.cd_dice = BinaryDICELoss()
        self.scd_focal = FocalLoss(ignore_index=0, alpha=opt.alpha, gamma=opt.gamma)
        self.scd_dice = DICELoss(ignore_index=0)

        self.optimizer = optim.AdamW(self.detector.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.schedular = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=445 * opt.warmup_epochs,
                                                         num_training_steps=445 * opt.num_epochs)

        if opt.load_pretrain:
            self.load_ckpt(self.detector, self.optimizer, opt.name, opt.backbone)
        self.detector.cuda()
        # self.detector = nn.DataParallel(self.detector, device_ids=opt.gpu_ids)
        print("---------- Networks initialized -------------")

    def forward(self, x1, x2, label1, label2, cd_label):
        pred, pred_seg1, pred_seg2, pred_p2, pred_p3, pred_p4, pred_p5 = self.detector(x1, x2)
        # label = label.unsqueeze(1).float()
        cd_label = cd_label.long()
        cd_focal = self.cd_focal(pred, cd_label)
        cd_dice = self.cd_dice(pred, cd_label)
        p2_loss = self.cd_focal(pred_p2, cd_label) * 0.5 + self.cd_dice(pred_p2, cd_label)
        p3_loss = self.cd_focal(pred_p3, cd_label) * 0.5 + self.cd_dice(pred_p3, cd_label)
        p4_loss = self.cd_focal(pred_p4, cd_label) * 0.5 + self.cd_dice(pred_p4, cd_label)
        p5_loss = self.cd_focal(pred_p5, cd_label) * 0.5 + self.cd_dice(pred_p5, cd_label)
        cd_loss = cd_focal * 0.5 + cd_dice + p3_loss + p4_loss + p5_loss
        seg_loss = self.scd_focal(pred_seg1, label1) * 0.5 + self.scd_dice(pred_seg1, label1) \
                   + self.scd_focal(pred_seg2, label2) * 0.5 + self.scd_dice(pred_seg2, label2)

        return cd_loss, seg_loss

    def inference(self, x1, x2):
        with torch.no_grad():
            pred, pred_seg1, pred_seg2, _, _, _, _ = self.detector(x1, x2)
        return pred, pred_seg1, pred_seg2

    def adjust_learning_rate(self, iter, total_iters, min_lr=1e-6, power=0.9):
        lr = (self.base_lr - min_lr) * (1 - iter / total_iters) ** power + min_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_ckpt(self, network, optimizer, name, backbone):
        save_filename = '%s_%s_best.pth' % (name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            print("%s not exists yet!" % save_path)
            raise ("%s must exist!" % save_filename)
        else:
            checkpoint = torch.load(save_path, map_location=self.device)
            network.load_state_dict(checkpoint['network'], False)

    def save_ckpt(self, network, optimizer, model_name, backbone):
        save_filename = '%s_%s_best.pth' % (model_name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path):
            os.remove(save_path)
        torch.save({'network': network.cpu().state_dict(),
                    'optimizer': optimizer.state_dict()},
                   save_path)
        if torch.cuda.is_available():
            network.cuda()

    def save(self, model_name, backbone):
        self.save_ckpt(self.detector, self.optimizer, model_name, backbone)

    def name(self):
        return self.opt.name


def create_model(opt):
    model = Model(opt)
    print("model [%s] was created" % model.name())

    return model.cuda()

