from .transform import Transforms
from util.palette import Color2Index
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def make_dataset(dir):
    img_paths = []
    names = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            img_paths.append(path)
            names.append(fname)

    return img_paths, names


class Load_Dataset(Dataset):
    def __init__(self, opt):
        super(Load_Dataset, self).__init__()
        self.opt = opt

        self.dir1 = os.path.join(opt.dataroot, opt.dataset, opt.phase, 'im1')
        self.t1_paths, self.fnames = sorted(make_dataset(self.dir1))

        self.dir2 = os.path.join(opt.dataroot, opt.dataset, opt.phase, 'im2')
        self.t2_paths, _ = sorted(make_dataset(self.dir2))

        self.dir_label1 = os.path.join(opt.dataroot, opt.dataset, opt.phase, 'label1')
        self.label1_paths, _ = sorted(make_dataset(self.dir_label1))

        self.dir_label2 = os.path.join(opt.dataroot, opt.dataset, opt.phase, 'label2')
        self.label2_paths, _ = sorted(make_dataset(self.dir_label2))

        self.dataset_size = len(self.t1_paths)

        self.normalize = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transform = transforms.Compose([Transforms()])
        self.to_tensor = transforms.Compose([transforms.ToTensor()])


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        t1_path = self.t1_paths[index]
        fname = self.fnames[index]
        img1 = Image.open(t1_path)

        t2_path = self.t2_paths[index]
        img2 = Image.open(t2_path)

        label1_path = self.label1_paths[index]
        label1 = Image.open(label1_path)
        label1 = Image.fromarray(Color2Index(self.opt.dataset, np.array(label1)))

        label2_path = self.label2_paths[index]
        label2 = Image.open(label2_path)
        label2 = Image.fromarray(Color2Index(self.opt.dataset, np.array(label2)))

        mask = np.array(label1)
        cd_label = np.ones_like(mask)
        cd_label[mask == 0] = 0
        cd_label = Image.fromarray(cd_label)

        if self.opt.phase == 'train':
            _data = self.transform(
                {'img1': img1, 'img2': img2, 'label1': label1, 'label2': label2, 'cd_label': cd_label})
            img1, img2, label1, label2, cd_label = _data['img1'], _data['img2'], _data['label1'], _data['label2'], \
                                                   _data['cd_label']

        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        label1 = torch.from_numpy(np.array(label1)).long()
        label2 = torch.from_numpy(np.array(label2)).long()
        cd_label = torch.from_numpy(np.array(cd_label)).long()
        input_dict = {'img1': img1, 'img2': img2, 'label1': label1, 'label2': label2, 'cd_label': cd_label,
                      'fname': fname}

        return input_dict


class DataLoader(torch.utils.data.Dataset):

    def __init__(self, opt):
        self.dataset = Load_Dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=opt.batch_size,
                                                      shuffle=opt.phase=='train',
                                                      pin_memory=True,
                                                      drop_last=opt.phase=='train',
                                                      num_workers=int(opt.num_workers)
                                                      )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
