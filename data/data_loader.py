import torch
from torch.utils import data
from torchvision import transforms
import os
from sklearn.datasets import fetch_openml

from PIL import Image
import numpy as np


def make_datapath_list(label_list=None):
    if label_list is None:
        label_list = [1, 4]
    train_img_list = []
    train_label_list = []
    for img_idx in range(200):
        for i, label in enumerate(label_list):
            img_path = f"./data/img_{label}/img_{label}_{img_idx}.png"
            train_img_list.append(img_path)
            train_label_list.append(i)

    return train_img_list, train_label_list


class ImageTransform:
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    def __call__(self, img):
        return self.data_transform(img)


class ImageDataset(data.Dataset):
    def __init__(self, data_list, transform, label_list=None):
        self.data_list = data_list
        self.transform = transform
        self.label = label_list
        self.label_mask = self._create_label_mask()

    def _create_label_mask(self):
        if self.label is not None:
            label_mask = np.zeros(len(self.label))
            num_label = len(set(self.label))
            label_mask[0:50*num_label] = 1
            np.random.shuffle(label_mask)
            label_mask = torch.LongTensor(label_mask)
            return label_mask
        return None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img)

        if self.label is not None:
            return img_transformed, self.label[index], self.label_mask[index]

        return img_transformed


data_dir = "/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

    mnist = fetch_openml('mnist_784', version=1, data_home="./data/", as_frame=False)

data_dir_path_1 = "./data/img_1/"
data_dir_path_2 = "./data/img_2/"
data_dir_path_3 = "./data/img_3/"
data_dir_path_4 = "./data/img_4/"
data_dir_path_5 = "./data/img_5/"
data_dir_path_6 = "./data/img_6/"
data_dir_path_7 = "./data/img_7/"
data_dir_path_8 = "./data/img_8/"
data_dir_path_9 = "./data/img_9/"
data_dir_path_0 = "./data/img_0/"
if not os.path.exists(data_dir_path_1):
    mnist = fetch_openml('mnist_784', version=1, data_home="./data/", as_frame=False)
    os.mkdir(data_dir_path_1)
    os.mkdir(data_dir_path_2)
    os.mkdir(data_dir_path_3)
    os.mkdir(data_dir_path_4)
    os.mkdir(data_dir_path_5)
    os.mkdir(data_dir_path_6)
    os.mkdir(data_dir_path_7)
    os.mkdir(data_dir_path_8)
    os.mkdir(data_dir_path_9)
    os.mkdir(data_dir_path_0)
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0
    count8 = 0
    count9 = 0
    count0 = 0
    max_num = 200

    X = mnist.data
    y = mnist.target
    for i in range(len(X)):

        if (y[i] == "1") and (count1 < max_num):
            file_path = "./data/img_1/img_1_" + str(count1) + ".png"
            im_f = (X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count1 += 1

        if (y[i] == "2") and (count2 < max_num):
            file_path = "./data/img_2/img_2_" + str(count2) + ".png"
            im_f = (X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count2 += 1

        if (y[i] == "3") and (count3 < max_num):
            file_path = "./data/img_3/img_3_" + str(count3) + ".png"
            im_f = (X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count3 += 1

        if (y[i] == "4") and (count4 < max_num):
            file_path = "./data/img_4/img_4_" + str(count4) + ".png"
            im_f = (X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count4 += 1

        if (y[i] == "5") and (count5 < max_num):
            file_path = "./data/img_5/img_5_" + str(count5) + ".png"
            im_f = (X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count5 += 1

        if (y[i] == "6") and (count6 < max_num):
            file_path = "./data/img_6/img_6_" + str(count6) + ".png"
            im_f = (X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count6 += 1

        # 画像7の作成
        if (y[i] == "7") and (count7 < max_num):
            file_path = "./data/img_7/img_7_" + str(count7) + ".png"
            im_f = (X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count7 += 1

            # 画像8の作成
        if (y[i] == "8") and (count8 < max_num):
            file_path = "./data/img_8/img_8_" + str(count8) + ".png"
            im_f = (X[i].reshape(28, 28))  # 画像を28*28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count8 += 1

        if (y[i] == "9") and (count9 < max_num):
            file_path = "./data/img_9/img_9_" + str(count9) + ".png"
            im_f = (X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count9 += 1

        if (y[i] == "0") and (count0 < max_num):
            file_path = "./data/img_0/img_0_" + str(count0) + ".png"
            im_f = (X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count0 += 1

if __name__ == "__main__":
    train_img_list, _ = make_datapath_list()
    mean = (0.5,)
    std = (0.5,)
    train_dataset = ImageDataset(file_list=train_img_list, transform=ImageTransform(mean, std))
    batch_size = 64
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    batch_iterator = iter(train_dataloader)
    imges = next(batch_iterator)
    print(imges.size())

