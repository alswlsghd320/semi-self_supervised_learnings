from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets.utils import download_url, check_integrity


class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    nclass = 10
    split_list = ['label', 'unlabel', 'valid', 'test', 'pseudo']

    def __init__(self, root, split='train',
                 transform=None, target_transform=None,
                 download=False, boundary=0, indices=None, labels=None, two_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.indices = indices
        self.labels = labels
        self.two_transform = two_transform
        assert (boundary < 10)
        print("Boundary: ", boundary)
        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.split == 'label' or self.split == 'unlabel' or self.split == 'valid' or self.split == 'pseudo':
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            if boundary != 0:
                bidx = 5000 * boundary
                self.train_data = [self.train_data[bidx:], self.train_data[:bidx]]
                self.train_data = np.concatenate(self.train_data)
                self.train_labels = [self.train_labels[bidx:], self.train_labels[:bidx]]
                self.train_labels = np.concatenate(self.train_labels)

            train_datau = []
            train_labelsu = []
            train_data1 = []
            train_labels1 = []
            valid_data1 = []
            valid_labels1 = []
            num_labels_valid = [0 for _ in range(self.nclass)]
            num_labels_train = [0 for _ in range(self.nclass)]
            for i in range(self.train_data.shape[0]):
                tmp_label = self.train_labels[i]
                if num_labels_valid[tmp_label] < 500:
                    valid_data1.append(self.train_data[i])
                    valid_labels1.append(self.train_labels[i])
                    num_labels_valid[tmp_label] += 1
                elif num_labels_train[tmp_label] < 400:
                    train_data1.append(self.train_data[i])
                    train_labels1.append(self.train_labels[i])
                    num_labels_train[tmp_label] += 1

                    # train_datau.append(self.train_data[i])
                    # train_labelsu.append(self.train_labels[i])
                else:
                    train_datau.append(self.train_data[i])
                    train_labelsu.append(self.train_labels[i])

            if self.split == 'label':
                self.train_data = train_data1
                self.train_labels = train_labels1

                self.train_data = np.concatenate(self.train_data)
                self.train_data = self.train_data.reshape((len(train_data1), 3, 32, 32))
                self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

                num_tr = self.train_data.shape[0]
                # print(self.train_data1[:1,:1,:5,:5])
                # print(self.train_labels1[:10])
                # print(self.train_data[:1,:1,:5,:5])
                # print(self.train_labels[:10])
                print('Label: ', num_tr)  # label

                # self.midx=0
                # self.idx_offset = num_tr_ul - (num_tr_ul//num_tr) * num_tr
                # print('Offset: :',self.idx_offset)

            elif self.split == 'unlabel':
                self.train_data_ul = train_datau
                self.train_labels_ul = train_labelsu

                self.train_data_ul = np.concatenate(self.train_data_ul)
                self.train_data_ul = self.train_data_ul.reshape((len(train_datau), 3, 32, 32))
                self.train_data_ul = self.train_data_ul.transpose((0, 2, 3, 1))  # convert to HWC

                num_tr_ul = self.train_data_ul.shape[0]
                print('Unlabel: ', num_tr_ul)  # unlabel

            elif self.split == 'valid':
                self.valid_data = valid_data1
                self.valid_labels = valid_labels1

                self.valid_data = np.concatenate(self.valid_data)
                self.valid_data = self.valid_data.reshape((len(valid_data1), 3, 32, 32))
                self.valid_data = self.valid_data.transpose((0, 2, 3, 1))  # convert to HWC

                num_val = self.valid_data.shape[0]
                print('Valid: ', num_val)  # valid
                # print(self.valid_data[:1,:1,:5,:5])
                # print(self.valid_labels[:10])

            elif self.split == 'pseudo':
                self.train_data_ul = list(np.array(train_datau)[self.indices])
                self.train_labels_ul = self.labels

                self.train_data_ul = np.concatenate(self.train_data_ul)

                self.train_data_ul = self.train_data_ul.reshape((len(self.indices), 3, 32, 32))
                self.train_data_ul = self.train_data_ul.transpose((0, 2, 3, 1))  # convert to HWC

                num_tr_ul = self.train_data_ul.shape[0]
                print('Pseudo: ', num_tr_ul)

        elif self.split == 'test':
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'label':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'unlabel':
            img, target = self.train_data_ul[index], self.train_labels_ul[index]
        elif self.split == 'valid':
            img, target = self.valid_data[index], self.valid_labels[index]
        elif self.split == 'test':
            img, target = self.test_data[index], self.test_labels[index]
        elif self.split == 'pseudo':
            img, target = self.train_data_ul[index], self.train_labels_ul[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img1 = np.copy(img)
        img = Image.fromarray(img)
        #img1 = Image.fromarray(img1)

        if self.transform is not None:
            img = self.transform(img)
            if self.two_transform is not None:
                img1 = Image.fromarray(img1)
                img1 = self.transform(img1)
                return img, img1, target

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.split == 'label':
            return len(self.train_data)
        elif self.split == 'unlabel':
            return len(self.train_data_ul)
        elif self.split == 'valid':
            return len(self.valid_data)
        elif self.split == 'test':
            return len(self.test_data)
        elif self.split == 'pseudo':
            return len(self.train_data_ul)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_augmentation(img_size=32, ver=1):
    if ver == 1:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif ver == 2: #weak augmentation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAffine(degrees=(10, 30)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif ver == 3: #Hard augmentation
        transform = transforms.Compose([
            transforms.ToTensor(),
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
    elif ver == 4:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=(10, 30)),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    return transform