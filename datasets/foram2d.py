import os
import glob
import random

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from natsort import natsorted
import torchvision.transforms.functional as F


class ForamDataset2D(data.Dataset):
    def __init__(self, root, phase='train', img_size=256, transform=False):
        self.root = root
        self.img_size = img_size
        self.transform = transform

        self.img_paths, self.label_paths = [], []
        if phase == 'train':
            img_paths_train = glob.glob(os.path.join(self.root + '/images_cropped/train', '*'))  # change to your own path
            label_paths_train = glob.glob(os.path.join(self.root + '/labels_cropped_binary/train', '*'))
            for path in img_paths_train:
                pngs = glob.glob(os.path.join(path, '*'))
                pngs = natsorted(pngs)
                self.img_paths.extend(pngs)
            for path in label_paths_train:
                pngs = glob.glob(os.path.join(path, '*'))
                pngs = natsorted(pngs)
                self.label_paths.extend(pngs)
        elif phase == 'val':
            img_paths_val = glob.glob(os.path.join(self.root + '/images_cropped/validation', '*'))
            label_paths_val = glob.glob(os.path.join(self.root + '/labels_cropped_binary/validation', '*'))
            for path in img_paths_val:
                pngs = glob.glob(os.path.join(path, '*'))
                pngs = natsorted(pngs)
                self.img_paths.extend(pngs)
            for path in label_paths_val:
                pngs = glob.glob(os.path.join(path, '*'))
                pngs = natsorted(pngs)
                self.label_paths.extend(pngs)
        elif phase == 'test':
            img_paths_test = glob.glob(os.path.join(self.root + '/images_cropped/test', '*'))
            label_paths_test = glob.glob(os.path.join(self.root + '/labels_cropped_binary/test', '*'))
            for path in img_paths_test:
                pngs = glob.glob(os.path.join(path, '*'))
                pngs = natsorted(pngs)
                self.img_paths.extend(pngs)
            for path in label_paths_test:
                pngs = glob.glob(os.path.join(path, '*'))
                pngs = natsorted(pngs)
                self.label_paths.extend(pngs)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img = Image.open(self.img_paths[i])
        label = Image.open(self.label_paths[i])
        img, label = self.resize(img), self.resize(label)

        if self.transform:
            if random.random() > 0.5:
                img = F.hflip(img)
                label = F.hflip(label)
            if random.random() > 0.5:
                img = F.vflip(img)
                label = F.vflip(label)

        img, label = img.unsqueeze(0), label.unsqueeze(0)

        if self.transform:
            angle = random.randint(-90, 90)
            img = F.rotate(img, angle)
            label = F.rotate(label, angle)

        img = self.normalize(img)
        filename = self.img_paths[i].split('/')[-2] + '/' + self.img_paths[i].split('/')[-1]

        return img, label, filename

    def resize(self, img):
        w, h = img.size
        ratio = self.img_size / max(w, h)
        scaled_img = F.resize(img=img, size=[int(w * ratio), int(h * ratio)], interpolation=F.InterpolationMode.NEAREST)

        output = torch.zeros((self.img_size, self.img_size), dtype=torch.float32)
        scaled_w, scaled_h = scaled_img.size
        output[:scaled_h, :scaled_w] = torch.tensor(np.array(scaled_img))

        return output

    def normalize(self, image):
        image_min = image.min()
        image_max = image.max()
        return (image - image_min) / (image_max - image_min)
