import os
import glob

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from natsort import natsorted
from skimage.transform import resize
import torchio as tio
from scipy.ndimage import distance_transform_edt


class ForamDataset3D(data.Dataset):
    def __init__(self, root, transform, phase='train', img_size=(64, 128, 128), seg='mtl'):
        self.root = root
        self.transform = transform
        self.img_size = img_size
        self.seg = seg

        self.img_paths, self.label_paths = [], []
        if phase == 'train':
            img_paths_train = glob.glob(os.path.join(self.root + '/images_cropped/train', '*'))  # change to your own path
            label_paths_train = glob.glob(os.path.join(self.root + '/labels_cropped_multi/train', '*'))
            self.img_paths = img_paths_train
            self.label_paths = label_paths_train
        elif phase == 'val':
            img_paths_val = glob.glob(os.path.join(self.root + '/images_cropped/val', '*'))
            label_paths_val = glob.glob(os.path.join(self.root + '/labels_cropped_multi/val', '*'))
            self.img_paths = img_paths_val
            self.label_paths = label_paths_val
        elif phase == 'test':
            img_paths_test = glob.glob(os.path.join(self.root + '/images_cropped/test', '*'))
            label_paths_test = glob.glob(os.path.join(self.root + '/labels_cropped_multi/test', '*'))
            self.img_paths = img_paths_test
            self.label_paths = label_paths_test

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        x = self._form_volume(self.img_paths[i], resize=True)  # (D, W, H)
        y = self._form_volume(self.label_paths[i], resize=True)  # (D, W, H)
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        if self.transform:
            subject = tio.Subject(
                x=tio.Image(tensor=x, type=tio.INTENSITY),
                y=tio.Image(tensor=y, type=tio.LABEL),
            )

            transformed_subject = self.transform(subject)

            x = transformed_subject['x'].tensor
            y = transformed_subject['y'].tensor

        x = x / 255.0

        if self.seg == 'unet':
            mask = y
        elif self.seg == 'mtl':
            label = y.numpy()
            binary_mask = (label > 0)
            distance = distance_transform_edt(binary_mask)
            bdry_mask = torch.tensor((distance > 0) & (distance <= 1), dtype=torch.float32)
            bg_mask = torch.tensor(label == 0, dtype=torch.float32)
            fg_mask = torch.ones_like(y, dtype=torch.float32)
            fg_mask = fg_mask - bg_mask - bdry_mask
            mask = (fg_mask, bdry_mask, bg_mask)
        elif self.seg == 'plantseg':
            binary_mask = (y > 0)
            distance = distance_transform_edt(binary_mask)
            mask = torch.tensor(((distance > 0) & (distance <= 1)), dtype=torch.float32)
        else:
            raise ValueError('Semantic segmentation method is not correct.')

        filename = os.path.basename(self.img_paths[i])

        return x, mask, filename

    def _form_volume(self, sample_path, resize=True):
        slice_paths = glob.glob(os.path.join(sample_path, '*'))
        slice_paths = natsorted(slice_paths)

        volume = []
        for slice_path in slice_paths:
            slice_img = Image.open(slice_path)
            volume.append(np.array(slice_img))
        volume = np.stack(volume, axis=0)
        if resize:
            volume = self._resize_volume_with_aspect_ratio(volume, self.img_size)
        return torch.tensor(volume, dtype=torch.float)

    def _resize_volume_with_aspect_ratio(self, volume, target_size):
        scale_depth = target_size[0] / volume.shape[0]
        scale_height = target_size[1] / volume.shape[1]
        scale_width = target_size[2] / volume.shape[2]

        # Resize the volume
        scale = min(scale_depth, scale_height, scale_width)
        new_depth = int(volume.shape[0] * scale)
        new_height = int(volume.shape[1] * scale)
        new_width = int(volume.shape[2] * scale)
        resized_volume = resize(
            volume,
            (new_depth, new_height, new_width),
            mode='constant',
            preserve_range=True,
            order=0
        )

        # Pad the volume to match the target size
        pad_depth = (target_size[0] - new_depth) // 2
        pad_height = (target_size[1] - new_height) // 2
        pad_width = (target_size[2] - new_width) // 2
        padded_volume = np.pad(
            resized_volume,
            pad_width=((pad_depth, target_size[0] - new_depth - pad_depth),
                       (pad_height, target_size[1] - new_height - pad_height),
                       (pad_width, target_size[2] - new_width - pad_width)),
            mode='constant',
            constant_values=0
        )
        return padded_volume
