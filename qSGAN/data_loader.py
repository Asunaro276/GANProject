import torch
from torch.utils import data
from torchvision import transforms
import os

import cv2
import numpy as np


def make_mnist_datapath_list(label_list, num_data):
    train_img_list = []
    train_label_list = []
    for label in label_list:
        dir_path = f"../data/img_{label}/"
        for path in os.listdir(dir_path):
            train_img_list.append(os.path.join(dir_path, path))
            train_label_list.append(label)

    return train_img_list, train_label_list


def make_01_datapath_list():
    train_img_list = []
    train_label_list = []
    dir_path = "../data/img_qSGAN/"
    for label in [0, 1]:
        dir_label_path = os.path.join(dir_path, f"img_{label}/")
        for path in os.listdir(dir_label_path):
            train_img_list.append(os.path.join(f"../data/img_qSGAN/img_{label}", path))
            train_label_list.append(label)

    return train_img_list, train_label_list


def make_bas_2_datapath_list():
    train_img_list = []
    train_label_list = []
    dir_path = "../data/img_bars_and_strips_2/"
    for i, label in enumerate(["b", "s"]):
        dir_label_path = os.path.join(dir_path, f"img_{label}/")
        for path in os.listdir(dir_label_path):
            train_img_list.append(os.path.join(f"../data/img_bars_and_strips_2/img_{label}", path))
            train_label_list.append(i)

    return train_img_list, train_label_list


def make_bas_3_datapath_list():
    train_img_list = []
    train_label_list = []
    dir_path = "../data/img_bars_and_strips_3/"
    for i, label in enumerate(["b", "s"]):
        dir_label_path = os.path.join(dir_path, f"img_{label}/")
        for path in os.listdir(dir_label_path):
            train_img_list.append(os.path.join(f"../data/img_bars_and_strips_3/img_{label}", path))
            train_label_list.append(i)

    return train_img_list, train_label_list


def make_bas_4_datapath_list():
    train_img_list = []
    train_label_list = []
    dir_path = "../data/img_bars_and_strips_4/"
    for i, label in enumerate(["b", "s"]):
        dir_label_path = os.path.join(dir_path, f"img_{label}/")
        for path in os.listdir(dir_label_path):
            train_img_list.append(os.path.join(f"../data/img_bars_and_strips_4/img_{label}", path))
            train_label_list.append(i)

    return train_img_list, train_label_list


class ImageTransform:
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    def __call__(self, img):
        return self.data_transform(img)


class MNISTTransform:
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(8),
            transforms.Normalize(mean, std),
        ])

    def __call__(self, img):
        return self.data_transform(img)


class ImageDataset(data.Dataset):
    def __init__(self, data_list, num_mask, transform, label_list=None):
        self.data_list = data_list
        self.transform = transform
        self.num_mask = num_mask
        self.label = label_list
        self.label_mask = self._create_label_mask()

    def _create_label_mask(self):
        if self.label is not None:
            label_mask = np.zeros(len(self.label))
            label_mask[0:self.num_mask] = 1
            np.random.shuffle(label_mask)
            label_mask = torch.LongTensor(label_mask)
            return label_mask
        return None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img_transformed = self.transform(img)[0]

        if self.label is not None:
            return img_transformed, self.label[index], self.label_mask[index]

        return img_transformed
