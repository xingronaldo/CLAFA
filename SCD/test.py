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

if __name__ == '__main__':
    opt = Options().parse()
    #opt.batch_size = 1
    opt.phase = 'test'
    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    test_size = len(test_loader)
    print("#testing images = %d" % test_size)

    opt.load_pretrain = True
    model = create_model(opt)

    tbar = tqdm(test_data)
    total_iters = test_size
    metric = IOUandSek(num_classes=7)

    model.eval()
    for i, _data in enumerate(tbar):
        cd_out, seg_out1, seg_out2 = model.inference(_data['img1'].cuda(), _data['img2'].cuda())
        # update metric
        val_target = _data['cd_label'].detach()
        cd_out = torch.argmax(cd_out.detach(), dim=1)
        # val_pred = torch.where(val_pred > 0.5, torch.ones_like(val_pred), torch.zeros_like(val_pred)).long()
        seg_out1 = torch.argmax(seg_out1, dim=1).cpu().numpy()
        seg_out2 = torch.argmax(seg_out2, dim=1).cpu().numpy()
        cd_out = cd_out.cpu().numpy().astype(np.uint8)
        seg_out1[cd_out == 0] = 0
        seg_out2[cd_out == 0] = 0

        if opt.save_mask:
            cmap = color_map(opt.dataset)
            for i in range(seg_out1.shape[0]):
                mask = Image.fromarray(seg_out1[i].astype(np.uint8), mode="P")
                mask.putpalette(cmap)
                os.makedirs(os.path.join(opt.result_dir, 'test', 'im1'), exist_ok=True)
                mask.save(os.path.join(opt.result_dir, 'test', 'im1', _data['fname'][i]))

                mask = Image.fromarray(seg_out2[i].astype(np.uint8), mode="P")
                mask.putpalette(cmap)
                os.makedirs(os.path.join(opt.result_dir, 'test', 'im2'), exist_ok=True)
                mask.save(os.path.join(opt.result_dir, 'test', 'im2', _data['fname'][i]))

        metric.add_batch(seg_out1, _data['label1'].numpy())
        metric.add_batch(seg_out2, _data['label2'].numpy())
        score, miou, sek = metric.evaluate()
        tbar.set_description("Score: %.3f, IOU: %.3f, SeK: %.3f" % (score * 100.0, miou * 100.0, sek * 100.0))

