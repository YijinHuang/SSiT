import pickle
import random

import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageOps
from torchvision.transforms import functional as F

from funcs import print_msg


def build_dataset(args):
    transform = data_transforms(args.input_size)
    datasets = generate_dataset_from_pickle(args.data_index, transform, args.dataset_ratio)
    return datasets


def generate_dataset_from_pickle(pkl, transform, ratio=1.0):
    data = pickle.load(open(pkl, 'rb'))
    if ratio < 1.0:
        random.shuffle(data)
        data = data[:int(len(data)*ratio)]
    print_msg('Number of training samples: {}'.format(len(data)))

    train_dataset = PairGenerator(data, transform)
    return train_dataset


def data_transforms(input_size):
    mean = [0.425753653049469, 0.29737451672554016, 0.21293757855892181]  # eyepacs mean
    std = [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]  # eyepacs std
    data_aug = {
        'brightness': 0.4,
        'contrast': 0.4,
        'saturation': 0.2,
        'hue': 0.1,
        'scale_stu': (0.08, 0.8),
        'scale_tea': (0.8, 1.0),
        'degrees': (-180, 180),
    }

    transform = TransformWithMask(input_size, mean, std, data_aug)
    return transform


class PairGenerator(Dataset):
    def __init__(self, imgs, transform=None):
        super(PairGenerator, self).__init__()
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = self.pil_loader(img_path)
        mask = self.npy_loader(mask_path)
        mask = Image.fromarray(np.uint8(mask*255))
        if self.transform is not None:
            img_stu, img_tea, mask_stu, mask_tea = self.transform(img, mask)

        return img_stu, img_tea, mask_stu, mask_tea

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def npy_loader(self, path):
        with open(path, 'rb') as f:
            img = np.load(f)
            return img


class TwoCropTransform():
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class BYOLTransform():
    def __init__(self, transform_stu, transform_tea):
        self.transform_stu = transform_stu
        self.transform_tea = transform_tea

    def __call__(self, x1, x2):
        return [self.transform_stu(x1), self.transform_tea(x2)]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


class TransformWithMask(object):
    def __init__(self, input_size, mean, std, data_aug):
        scale_stu = data_aug['scale_stu']
        scale_tea = data_aug['scale_tea']
        jitter_param = (data_aug['brightness'], data_aug['contrast'], data_aug['saturation'], data_aug['hue'])
        degree = data_aug['degrees']

        self.resized_crop_stu = transforms.RandomResizedCrop(input_size, scale=scale_stu)
        self.color_jitter_stu = transforms.RandomApply([transforms.ColorJitter(*jitter_param)], p=0.8)
        self.grayscale_stu = transforms.RandomGrayscale(p=0.2)
        self.gaussian_blur_stu = transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0)
        self.rotation_stu = transforms.RandomRotation(degree)
        self.p_rotation_stu = 0.8
        self.p_hflip_stu = 0.5
        self.p_vflip_stu = 0.5

        self.resized_crop_tea = transforms.RandomResizedCrop(input_size, scale=scale_tea)
        self.color_jitter_tea = transforms.RandomApply([transforms.ColorJitter(*jitter_param)], p=0.8)
        self.grayscale_tea = transforms.RandomGrayscale(p=0.2)
        self.gaussian_blur_tea = transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1)
        self.rotation_tea = transforms.RandomRotation(degree)
        self.solarize_tea = transforms.RandomApply([Solarize()], p=0.2)
        self.p_rotation_tea = 0.8
        self.p_hflip_tea = 0.5
        self.p_vflip_tea = 0.5

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, img, mask):
        img_stu, mask_stu = self.resized_crop_with_mask(self.resized_crop_stu, img, mask)
        img_stu = self.color_jitter_stu(img_stu)
        img_stu = self.grayscale_stu(img_stu)
        img_stu = self.gaussian_blur_stu(img_stu)
        img_stu, mask_stu = self.rotation_with_mask(self.rotation_stu, img_stu, mask_stu, self.p_rotation_stu)
        img_stu, mask_stu = self.horizontal_flip_with_mask(img_stu, mask_stu, self.p_hflip_stu)
        img_stu, mask_stu = self.vertical_flip_with_mask(img_stu, mask_stu, self.p_vflip_stu)
        img_stu, mask_stu = self.to_tensor(img_stu), self.to_tensor(mask_stu)
        img_stu = self.normalize(img_stu)

        img_tea, mask_tea = self.resized_crop_with_mask(self.resized_crop_tea, img, mask)
        img_tea = self.color_jitter_tea(img_tea)
        img_tea = self.grayscale_tea(img_tea)
        img_tea = self.gaussian_blur_tea(img_tea)
        img_tea = self.solarize_tea(img_tea)
        img_tea, mask_tea = self.rotation_with_mask(self.rotation_tea, img_tea, mask_tea, self.p_rotation_tea)
        img_tea, mask_tea = self.horizontal_flip_with_mask(img_tea, mask_tea, self.p_hflip_tea)
        img_tea, mask_tea = self.vertical_flip_with_mask(img_tea, mask_tea, self.p_vflip_tea)
        img_tea, mask_tea = self.to_tensor(img_tea), self.to_tensor(mask_tea)
        img_tea = self.normalize(img_tea)

        return img_stu, img_tea, mask_stu, mask_tea

    def resized_crop_with_mask(self, tf, img, mask):
        assert isinstance(tf, transforms.RandomResizedCrop)
        i, j, h, w = tf.get_params(img, tf.scale, tf.ratio)
        img = F.resized_crop(img, i, j, h, w, tf.size, tf.interpolation)
        mask = F.resized_crop(mask, i, j, h, w, tf.size, tf.interpolation)
        return img, mask

    def rotation_with_mask(self, tf, img, mask, p):
        assert isinstance(tf, transforms.RandomRotation)
        if random.random() < p:
            angle = tf.get_params(tf.degrees)
            img = F.rotate(img, angle, tf.resample, tf.expand, tf.center, tf.fill)
            mask = F.rotate(mask, angle, tf.resample, tf.expand, tf.center, tf.fill)
        return img, mask

    def horizontal_flip_with_mask(self, img, mask, p):
        if random.random() < p:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask

    def vertical_flip_with_mask(self, img, mask, p):
        if random.random() < p:
            img = F.vflip(img)
            mask = F.vflip(mask)
        return img, mask
