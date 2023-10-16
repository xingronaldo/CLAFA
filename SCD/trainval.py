import torch
from option import Options
from data.cd_dataset import DataLoader
from model.create_model import create_model
from tqdm import tqdm
import math
from util.palette import color_map
from util.metric import IOUandSek
import os
import numpy as np
import random
from PIL import Image


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


class Trainval(object):
    def __init__(self, opt):
        self.opt = opt

        train_loader = DataLoader(opt)
        self.train_data = train_loader.load_data()
        train_size = len(train_loader)
        print("#training images = %d" % train_size)
        opt.phase = 'val'
        val_loader = DataLoader(opt)
        self.val_data = val_loader.load_data()
        val_size = len(val_loader)
        print("#validation images = %d" % val_size)
        opt.phase = 'train'

        self.model = create_model(opt)
        self.optimizer = self.model.optimizer
        self.schedular = self.model.schedular

        self.iters = 0
        self.total_iters = math.ceil(train_size / opt.batch_size) * opt.num_epochs
        self.previous_best = 0.0

    def train(self):
        tbar = tqdm(self.train_data)
        opt.phase = 'train'
        _loss = 0.0
        _cd_loss = 0.0
        _seg_loss = 0.0

        for i, data in enumerate(tbar):
            self.model.detector.train()
            cd_loss, seg_loss = self.model(data['img1'].cuda(), data['img2'].cuda(), data['label1'].cuda(),
                                           data['label2'].cuda(), data['cd_label'].cuda())
            loss = 2 * cd_loss + seg_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.schedular.step()
            _loss += loss.item()
            _cd_loss += cd_loss.item()
            _seg_loss += seg_loss.item()
            del loss

            #self.iters += 1
            #self.model.adjust_learning_rate(self.iters, self.total_iters)

            tbar.set_description("Loss: %.3f, CD: %.3f, Seg: %.3f, LR: %.6f" %
                                 (_loss / (i + 1), _cd_loss / (i + 1), _seg_loss / (i + 1), self.optimizer.param_groups[0]['lr']))

    def val(self):
        tbar = tqdm(self.val_data)
        metric = IOUandSek(num_classes=7)
        opt.phase = 'val'
        self.model.eval()

        with torch.no_grad():
            for i, _data in enumerate(tbar):
                cd_out, seg_out1, seg_out2 = self.model.inference(_data['img1'].cuda(), _data['img2'].cuda())
                # update metric
                val_target = _data['cd_label'].detach()
                cd_out = torch.argmax(cd_out.detach(), dim=1)
                #val_pred = torch.where(val_pred > 0.5, torch.ones_like(val_pred), torch.zeros_like(val_pred)).long()
                seg_out1 = torch.argmax(seg_out1, dim=1).cpu().numpy()
                seg_out2 = torch.argmax(seg_out2, dim=1).cpu().numpy()
                cd_out = cd_out.cpu().numpy().astype(np.uint8)
                seg_out1[cd_out == 0] = 0
                seg_out2[cd_out == 0] = 0

                if self.opt.save_mask:
                    cmap = color_map(self.opt.dataset)
                    for i in range(seg_out1.shape[0]):
                        mask = Image.fromarray(seg_out1[i].astype(np.uint8), mode="P")
                        mask.putpalette(cmap)
                        os.makedirs(os.path.join(self.opt.result_dir, 'val', 'im1'), exist_ok=True)
                        mask.save(os.path.join(self.opt.result_dir, 'val', 'im1', _data['fname'][i]))

                        mask = Image.fromarray(seg_out2[i].astype(np.uint8), mode="P")
                        mask.putpalette(cmap)
                        os.makedirs(os.path.join(self.opt.result_dir, 'val', 'im2'), exist_ok=True)
                        mask.save(os.path.join(self.opt.result_dir, 'val', 'im2', _data['fname'][i]))

                metric.add_batch(seg_out1, _data['label1'].numpy())
                metric.add_batch(seg_out2, _data['label2'].numpy())
                score, miou, sek = metric.evaluate()
                tbar.set_description("Score: %.2f, IOU: %.2f, SeK: %.2f" % (score * 100.0, miou * 100.0, sek * 100.0))

        if score >= self.previous_best:
            self.model.save(self.opt.name, self.opt.backbone)
            self.previous_best = score


if __name__ == "__main__":
    opt = Options().parse()
    trainval = Trainval(opt)
    setup_seed(seed=1)

    for epoch in range(1, opt.num_epochs + 1):
        print("\n==> Epoch %i, previous best = %.3f" % (epoch, trainval.previous_best * 100))
        trainval.train()
        trainval.val()



