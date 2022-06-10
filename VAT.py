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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Network
from utils import accuracy, VATLoss
from config import getConfig
from datasets.loader_cifar import CIFAR10, get_augmentation

import warnings
warnings.filterwarnings('ignore')

class Trainer():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
        self.model = Network(args).to(self.device)

        self.train_ds = CIFAR10(args.data_path, split='label', download=True, transform=get_augmentation(ver=2), boundary=0)
        self.unlabel_ds = CIFAR10(args.data_path, split='unlabel', download=True, transform=get_augmentation(ver=2), boundary=0)
        self.val_ds = CIFAR10(args.data_path, split='valid', download=True, transform=get_augmentation(ver=1), boundary=0)
        self.test_ds = CIFAR10(args.data_path, split='test', download=True, transform=get_augmentation(ver=1), boundary=0)

        self.train_dl = DataLoader(self.train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        self.unlabel_dl = DataLoader(self.unlabel_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        self.val_dl = DataLoader(self.val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        self.test_dl = DataLoader(self.test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        self.save_path = args.save_path
        self.writer = SummaryWriter(self.save_path) if args.use_tensorboard else None
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        self.vadv_loss = VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)
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

            train_loss, train_top1, train_top5 = self.train_one_epoch()
            val_loss, val_top1, val_top5 = self.validate()

            if self.writer is not None:
                self.writer.add_scalar('Train/top1_accuracy', train_top1, epoch)
                self.writer.add_scalar('Train/top5_accuracy', train_top5, epoch)
                self.writer.add_scalar('Train/loss', train_loss, epoch)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('Val/top1_accuracy', val_top1, epoch)
                self.writer.add_scalar('Val/top5_accuracy', val_top5, epoch)
                self.writer.add_scalar('Val/loss', val_loss, epoch)

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
                self.writer.add_scalar('Best/top1_accuracy', best_acc, epoch)
                self.writer.add_scalar('Best/top5_accuracy', best_acc2, epoch)
                self.writer.add_scalar('Best/loss', best_loss, epoch)

        end = time.time()
        print(f'Best Epoch:{best_epoch} | Loss:{best_loss:.4f} | Top1:{best_acc:.4f} | Top5:{best_acc2:.4f}')
        print(f'Total Training time:{(end - start) / 60:.3f}Minute')

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0
        top1 = 0
        top5 = 0

        labeled_iter = iter(self.train_dl)
        for images_us, _ in tqdm(self.unlabel_dl):
            images_us = torch.tensor(images_us, device=self.device, dtype=torch.float32)

            #Get labeled data
            try:
                images_l, targets = labeled_iter.next()
            except:
                self.scheduler.step()
                labeled_iter = iter(self.train_dl)
                images_l, targets = labeled_iter.next()

            images_l = torch.tensor(images_l, device=self.device, dtype=torch.float32)
            targets_l = torch.tensor(targets, device=self.device, dtype=torch.long)

            self.model.zero_grad(set_to_none=True)

            preds_l = self.model(images_l)
            loss_l = self.criterion(preds_l, targets_l) # labeled
            loss_vadv = self.vadv_loss(self.model, images_us) # unlabeld

            loss = loss_l + self.args.lambda_u * loss_vadv
            loss.backward()

            self.optimizer.step()

            t1, t5 = accuracy(preds_l, targets_l, (1, 5))
            train_loss += loss.item()
            top1 += t1
            top5 += t5

        top1 /= len(self.unlabel_dl)
        top5 /= len(self.unlabel_dl)
        train_loss /= len(self.unlabel_dl)

        return train_loss, top1, top5

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            top1 = 0
            top5 = 0

            for images, targets in tqdm(self.val_dl):
                images = torch.tensor(images, device=self.device, dtype=torch.float32)
                targets = torch.tensor(targets, device=self.device, dtype=torch.long)

                preds = self.model(images)
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
    trainer.train()