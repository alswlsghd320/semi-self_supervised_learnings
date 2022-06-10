import random
from os.path import join as opj

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

# 테스트셋도 레이블 알고있다고 가정
class Custom_Dataset(Dataset):
    def __init__(self, df, transform=None, two_transform=None):
        self.id = df['file_name'].values
        self.target = df['label'].values
        self.transform = transform

        print(f'Dataset size:{len(self.id)}')

    def __getitem__(self, idx):
        img_path = opj(self.data_path, self.id[idx])
        image = cv2.imread(img_path).astype(np.float32)
        target = self.target[idx]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        if self.transform is not None:
            img = self.transform(torch.from_numpy(image.transpose(2, 0, 1)))

        if self.two_transform is not None:
            img2 = self.two_transform(torch.from_numpy(image.transpose(2, 0, 1)))
            return img, img2, target

        return img, target

    def __len__(self):
        return len(self.id)

class GammaTransform:
    """Rotate by one of the given angles."""

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        gam = random.uniform(self.a, self.b)
        return TF.adjust_gamma(x, gam)

def get_train_augmentation(img_size, ver):
    if ver == 1:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif ver == 2:
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=(10, 30)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif ver == 3:
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=180),
            transforms.RandomAffine(degrees=(10, 30)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif ver == 4:
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=(10, 30)),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif ver == 5:
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=180),
            transforms.RandomAffine(degrees=(10, 30)),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif ver == 6:
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=180),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(30),
            transforms.ColorJitter(),
            transforms.RandomInvert(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif ver == 7:
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=180),
            transforms.RandomAffine(degrees=45),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif ver == 8:
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(30),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif ver == 9:
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(30),
            GammaTransform(0.6, 1.0),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif ver == 10:
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=(180)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(30),
            transforms.ColorJitter(),
            transforms.RandomInvert(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif ver == 11:
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transform