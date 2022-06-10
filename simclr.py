import os
import time
import random
import pprint
from os.path import join as opj

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch_optimizer as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from config import getConfig
from datasets.loader_cifar import CIFAR10, get_augmentation

import warnings
warnings.filterwarnings('ignore')

class Trainer():
    def __init__(self, args):
        self.args = args
        assert args.feature_dim != 0
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

        self.model = Network(args).to(self.device)
        self.classifier = Classification_Head(args).to(self.device)

        self.train_ds = CIFAR10(args.data_path, split='label', download=True, transform=get_augmentation(ver=2),
                           boundary=0, two_transform=get_augmentation(ver=3)) + \
                   CIFAR10(args.data_path, split='unlabel', download=True, transform=get_augmentation(ver=2),
                           boundary=0, two_transform=get_augmentation(ver=3))
        self.finetune_ds = CIFAR10(args.data_path, split='label', download=True, transform=get_augmentation(ver=2),
                                boundary=0)
        self.val_ds = CIFAR10(args.data_path, split='valid', download=True, transform=get_augmentation(ver=1), boundary=0, two_transform=get_augmentation(ver=4))
        self.test_ds = CIFAR10(args.data_path, split='test', download=True, transform=get_augmentation(ver=1), boundary=0)

        self.train_dl = DataLoader(self.train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        self.finetune_dl = DataLoader(self.finetune_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        self.val_dl = DataLoader(self.val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        self.test_dl = DataLoader(self.test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        self.save_path = args.save_path
        self.writer = SummaryWriter(self.save_path) if args.use_tensorboard else None
        self.criterion = SupConLoss(temperature=args.temperature)

    def init_settings(self, args):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        if args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestone,
                                                                  gamma=args.lr_factor, verbose=False)
        elif args.scheduler == 'cos':
            tmax = args.tmax  # half-cycle
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=tmax, eta_min=args.min_lr,
                                                                        verbose=False)
        elif args.scheduler == 'cycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.max_lr,
                                                              steps_per_epoch=iter_per_epoch, epochs=args.epochs)

    def close_writer(self):
        if self.writer is not None:
            self.writer.close()

    def train(self):
        self.init_settings(self.args)
        start = time.time()

        # Early stopping
        best_epoch = 0
        best_loss = np.inf
        early_stopping = 0

        for epoch in range(self.args.epochs):

            if self.args.scheduler == 'cos':
                if epoch > self.args.warm_epoch:
                    self.scheduler.step()

            train_loss = self.train_one_epoch()
            val_loss = self.validate(False)

            if self.writer is not None:
                self.writer.add_scalar('Train/loss', train_loss, epoch)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('Val/loss', val_loss, epoch)

            print(f'Epoch : {epoch} | Train Loss:{train_loss:.4f} | Val Loss:{val_loss:.4f}')
            state_dict = self.model.state_dict()

            if val_loss < best_loss:
                early_stopping = 0
                best_epoch = epoch
                best_loss = val_loss

                torch.save({'epoch': epoch,
                            'state_dict': state_dict,
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                            }, os.path.join(self.save_path, 'best_model.pth'))
            else:
                early_stopping += 1

            if early_stopping == self.args.patience:
                break

            if self.writer is not None:
                self.writer.add_scalar('Best/mse', best_loss, epoch)

        end = time.time()
        print(f'Best Epoch:{best_epoch} | Best Loss:{best_loss:.4f}')
        print(f'Total Training time:{(end - start) / 60:.3f}Minute')

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0

        for images1, images2, _ in tqdm(self.train_dl):
            images1 = torch.tensor(images1, device=self.device, dtype=torch.float32)
            images2 = torch.tensor(images2, device=self.device, dtype=torch.float32)

            self.model.zero_grad(set_to_none=True)

            f1 = F.normalize(self.model(images1), dim=1)
            f2 = F.normalize(self.model(images2), dim=1)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            loss = self.criterion(features) # Difference with SupCon(레이블 정보가 안들어감)
            loss.backward()

            self.optimizer.step()

            train_loss += loss.item()

        train_loss /= len(self.train_dl)

        return train_loss

    def finetune(self):
        self.model = nn.Sequential(self.model, self.classifier).to(self.device)
        self.val_ds = CIFAR10(self.args.data_path, split='valid', download=True, transform=get_augmentation(ver=1),
                              boundary=0)
        self.val_dl = DataLoader(self.val_ds, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        self.criterion = nn.CrossEntropyLoss()
        self.args.lr /= 2
        self.init_settings(self.args)
        start = time.time()

        # Early stopping
        best_epoch = 0
        best_loss = np.inf
        best_acc = 0
        best_acc2 = 0
        early_stopping = 0

        for epoch in range(self.args.epochs):

            if self.args.scheduler == 'cos':
                if epoch > self.args.warm_epoch:
                    self.scheduler.step()

            train_loss, train_top1, train_top5 = self.finetune_one_epoch()
            val_loss, val_top1, val_top5 = self.validate(True)

            if self.writer is not None:
                self.writer.add_scalar('Finetune/Train/top1_accuracy', train_top1, epoch)
                self.writer.add_scalar('Finetune/Train/top5_accuracy', train_top5, epoch)
                self.writer.add_scalar('Finetune/Train/loss', train_loss, epoch)
                self.writer.add_scalar('Finetune/Train/LR', self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('Finetune/Val/top1_accuracy', val_top1, epoch)
                self.writer.add_scalar('Finetune/Val/top5_accuracy', val_top5, epoch)
                self.writer.add_scalar('Finetune/Val/loss', val_loss, epoch)

            print(f'Epoch : {epoch} | Train Loss:{train_loss:.4f} | Train Top1:{train_top1:.4f} | Train Top5:{train_top5:.4f}')
            print(f'Epoch : {epoch} | Val Loss:{val_loss:.4f}   | Val Top1:{val_top1:.4f}   | Val Top5:{val_top5:.4f}')
            state_dict = self.model.state_dict()

            if val_top1 > best_acc:
                early_stopping = 0
                best_epoch = epoch
                best_loss = val_loss
                best_acc = val_top1
                best_acc2 = val_top5

                torch.save({'epoch' :epoch,
                            'state_dict' :state_dict,
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                            }, os.path.join(self.save_path, 'best_model.pth'))
            else:
                early_stopping += 1

            if early_stopping == self.args.patience:
                break

            if self.writer is not None:
                self.writer.add_scalar('Finetune/Best/top1_accuracy', best_acc, epoch)
                self.writer.add_scalar('Finetune/Best/top5_accuracy', best_acc2, epoch)
                self.writer.add_scalar('Finetune/Best/loss', best_loss, epoch)

        end = time.time()
        print(f'Best Epoch:{best_epoch} | Loss:{best_loss:.4f} | Top1:{best_acc:.4f} | Top5:{best_acc2:.4f}')
        print(f'Total Training time:{(end - start) / 60:.3f}Minute')

    def finetune_one_epoch(self):
        self.model.train()
        train_loss = 0
        top1 = 0
        top5 = 0

        for images, targets in tqdm(self.finetune_dl):
            images = torch.tensor(images, device=self.device, dtype=torch.float32)
            targets = torch.tensor(targets, device=self.device, dtype=torch.long)

            self.model.zero_grad(set_to_none=True)

            preds = self.model(images)

            loss = self.criterion(preds, targets)
            loss.backward()

            self.optimizer.step()

            t1, t5 = accuracy(preds, targets, (1, 5))
            train_loss += loss.item()
            top1 += t1
            top5 += t5

        top1 /= len(self.finetune_dl)
        top5 /= len(self.finetune_dl)
        train_loss /= len(self.finetune_dl)

        return train_loss, top1, top5 #train_acc

    def validate(self, finetune=False):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            top1 = 0
            top5 = 0

            if finetune:
                for images, targets in tqdm(self.val_dl):
                    images = torch.tensor(images, device=self.device, dtype=torch.float32)
                    targets = torch.tensor(targets, device=self.device, dtype=torch.long)

                    preds = F.normalize(self.model(images), dim=1)
                    loss = self.criterion(preds, targets)

                    # Metric
                    t1, t5 = accuracy(preds, targets, (1, 5))
                    val_loss += loss.item()
                    top1 += t1
                    top5 += t5

                top1 /= len(self.val_dl)
                top5 /= len(self.val_dl)
                val_loss /= len(self.val_dl)
                return val_loss, top1, top5
            else:
                for images1, images2, _ in tqdm(self.val_dl):
                    images1 = torch.tensor(images1, device=self.device, dtype=torch.float32)
                    images2 = torch.tensor(images2, device=self.device, dtype=torch.float32)

                    f1 = F.normalize(self.model(images1), dim=1)
                    f2 = F.normalize(self.model(images2), dim=1)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    loss = self.criterion(features)  # Difference with SupCon(레이블 정보가 안들어감)

                    # Metric
                    val_loss += loss.item()

                val_loss /= len(self.val_dl)
                return val_loss

    def test(self):
        pass

if __name__ == '__main__':
    args = getConfig()

    print('<---- Training Params ---->')
    pprint.pprint(args)

    # Random Seed
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.save_path, exist_ok=True)

    trainer = Trainer(args)
    trainer.train() # Feature extractor 학습
    trainer.finetune() # classifier 학습