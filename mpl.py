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

from model import *
from utils import *
from config import getConfig
from datasets.loader_cifar import CIFAR10, get_augmentation

import warnings
warnings.filterwarnings('ignore')

class MPL():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
        self.student = Network(args).to(self.device)
        self.teacher = Network(args).to(self.device)

        self.train_ds = CIFAR10(args.data_path, split='label', download=True, transform=get_augmentation(ver=2), boundary=0)
        self.unlabel_ds = CIFAR10(args.data_path, split='unlabel', download=True, transform=get_augmentation(ver=2), boundary=0, two_transform = get_augmentation(ver=3))
        self.val_ds = CIFAR10(args.data_path, split='valid', download=True, transform=get_augmentation(ver=1), boundary=0)
        self.test_ds = CIFAR10(args.data_path, split='test', download=True, transform=get_augmentation(ver=1), boundary=0)

        self.train_dl = DataLoader(self.train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        self.unlabel_dl = DataLoader(self.unlabel_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        self.val_dl = DataLoader(self.val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        self.test_dl = DataLoader(self.test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        self.save_path = args.save_path
        self.writer = SummaryWriter(self.save_path) if args.use_tensorboard else None

    def close_writer(self):
        if self.writer is not None:
            self.writer.close()

    def init_setting(self, args):
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        self.optimizer_s = torch.optim.AdamW(self.student.parameters(), lr=args.lr)
        self.optimizer_t = torch.optim.AdamW(self.teacher.parameters(), lr=args.lr)

        if args.scheduler == 'step':
            self.scheduler_s = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_s, milestones=args.milestone,
                                                                  gamma=args.lr_factor, verbose=False)
            self.scheduler_t = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_t, milestones=args.milestone,
                                                                  gamma=args.lr_factor, verbose=False)
        elif args.scheduler == 'cos':
            tmax = args.tmax  # half-cycle
            self.scheduler_s = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_s, T_max=tmax, eta_min=args.min_lr,
                                                                          verbose=False)
            self.scheduler_t = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_t, T_max=tmax, eta_min=args.min_lr,
                                                                          verbose=False)
        elif args.scheduler == 'cycle':
            self.scheduler_s = torch.optim.lr_scheduler.OneCycleLR(self.optimizer_s, max_lr=args.max_lr,
                                                                 steps_per_epoch=iter_per_epoch, epochs=args.epochs)
            self.scheduler_t = torch.optim.lr_scheduler.OneCycleLR(self.optimizer_t, max_lr=args.max_lr,
                                                                 steps_per_epoch=iter_per_epoch, epochs=args.epochs)

    def train(self):
        self.init_setting(self.args)
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
                    self.scheduler_t.step()

            t_losses, t_udas, t_mpls, s_losses = self.train_one_epoch()
            val_loss, val_top1, val_top5 = self.validate()

            if self.writer is not None:
                self.writer.add_scalar('Train/teacher/Loss_total', t_losses, epoch)
                self.writer.add_scalar('Train/teacher/Loss_uda', t_udas, epoch)
                self.writer.add_scalar('Train/teacher/Loss_mpl', t_mpls, epoch)
                self.writer.add_scalar('Train/teacher/LR_t', self.optimizer_t.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('Train/student/Loss', s_losses, epoch)
                self.writer.add_scalar('Train/student/LR_s', self.optimizer_s.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('Val/top1_accuracy', val_top1, epoch)
                self.writer.add_scalar('Val/top5_accuracy', val_top5, epoch)
                self.writer.add_scalar('Val/loss', val_loss, epoch)

            print(
                f'Epoch : {epoch} | Teacher Loss_total:{t_losses:.4f} | Teacher Loss_uda:{t_udas:.4f} | Teacher Loss_mpl:{t_mpls:.4f} | Student Loss_total:{s_losses:.4f}')
            print(f'Epoch : {epoch} | Val Loss:{val_loss:.4f}   | Val Top1:{val_top1:.4f}   | Val Top5:{val_top5:.4f}')
            state_dict = self.student.state_dict()
            state_dict_t = self.teacher.state_dict()

            if val_loss < best_loss:
                early_stopping = 0
                best_epoch = epoch
                best_loss = val_loss
                best_acc = val_top1
                best_acc2 = val_top5

                torch.save({'epoch': epoch,
                            'state_dict': state_dict,
                            'state_dict_t': state_dict_t,
                            'optimizer': self.optimizer_s.state_dict(),
                            'scheduler': self.scheduler_s.state_dict(),
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
        self.student.train()
        self.teacher.train()

        t_losses = 0
        t_udas = 0
        t_mpls = 0
        s_losses = 0

        labeled_iter = iter(self.train_dl)
        for images_us, images_uh, _ in tqdm(self.unlabel_dl):
            images_us = torch.tensor(images_us, device=self.device, dtype=torch.float32)
            images_uh = torch.tensor(images_uh, device=self.device, dtype=torch.float32)

            #Get labeled data
            try:
                images_l, targets = labeled_iter.next()
            except:
                self.scheduler_s.step()
                labeled_iter = iter(self.train_dl)
                images_l, targets = labeled_iter.next()

            images_l = torch.tensor(images_l, device=self.device, dtype=torch.float32)
            targets = torch.tensor(targets, device=self.device, dtype=torch.long)

            self.teacher.zero_grad(set_to_none=True)
            self.student.zero_grad(set_to_none=True)

            t_images = torch.cat([images_l, images_us, images_uh])
            t_preds = self.teacher(t_images)

            t_preds_l = t_preds[:self.args.batch_size]  # preds for only labelset
            t_preds_us, t_preds_uh = t_preds[self.args.batch_size:].chunk(2)
            del t_preds

            # Teacher는 레이블된 데이터로 학습
            t_loss_l = self.criterion(t_preds_l, targets)

            pseudo_label = torch.softmax(t_preds_us.detach() / self.args.temperature, dim=-1)
            max_probs, soft_pseudo_label = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.args.threshold).float()

            t_loss_u = torch.mean(-(pseudo_label * torch.log_softmax(t_preds_uh, dim=-1)).sum(dim=-1) * mask)
            weight_u = self.args.lambda_u
            t_loss_uda = t_loss_l + weight_u * t_loss_u

            # Student는 labelset과 hard_transfrom된 이미지 사용
            s_images = torch.cat([images_l, images_uh])
            s_preds = self.student(s_images)
            s_preds_l = s_preds[:self.args.batch_size]
            s_preds_uh = s_preds[self.args.batch_size:]
            del s_preds

            s_loss_l_old = F.cross_entropy(s_preds_l.detach(), targets)  # Teacher가 학습하기 위한 loss 계산(학습 전)
            s_loss = self.criterion(s_preds_uh, soft_pseudo_label)  # Student는 Teacher로 인한 soft_psuedo_label를 이용하여 학습
            s_loss.backward()

            with torch.no_grad():
                s_preds_l = self.student(images_l)

            s_loss_l_new = F.cross_entropy(s_preds_l.detach(), targets)  # Teacher가 학습하기 위한 loss 계산(학습 후)

            # Student의 labelset에 대한 학습 전과 학습 후의 loss 차이
            dot_product = s_loss_l_new - s_loss_l_old
            _, hard_pseudo_label = torch.max(t_preds_uh.detach(), dim=-1)

            t_loss_mpl = dot_product * F.cross_entropy(t_preds_uh, hard_pseudo_label)

            t_loss = t_loss_uda + t_loss_mpl
            t_loss.backward()

            self.optimizer_s.step()
            self.optimizer_t.step()

            t_losses += t_loss.item()
            t_udas += t_loss_uda.item()
            t_mpls += t_loss_mpl.item()
            s_losses += s_loss.item()

        t_losses /= len(self.unlabel_dl)
        t_udas /= len(self.unlabel_dl)
        t_mpls /= len(self.unlabel_dl)
        s_losses /= len(self.unlabel_dl)

        return t_losses, t_udas, t_mpls, s_losses

    def finetune(self):
        self.args.lr /= 2
        self.init_setting(self.args)
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
                    self.scheduler_s.step()

            train_loss, train_top1, train_top5 = self.finetune_one_epoch()
            val_loss, val_top1, val_top5 = self.validate()

            if self.writer is not None:
                self.writer.add_scalar('Train/top1_accuracy', train_top1, epoch)
                self.writer.add_scalar('Train/top5_accuracy', train_top5, epoch)
                self.writer.add_scalar('Train/loss', train_loss, epoch)
                self.writer.add_scalar('Train/LR', self.optimizer_s.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('Val/top1_accuracy', val_top1, epoch)
                self.writer.add_scalar('Val/top5_accuracy', val_top5, epoch)
                self.writer.add_scalar('Val/loss', val_loss, epoch)

            print(f'Epoch : {epoch} | Train Loss:{train_loss:.4f} | Train Top1:{train_top1:.4f} | Train Top5:{train_top5:.4f}')
            print(f'Epoch : {epoch} | Val Loss:{val_loss:.4f}   | Val Top1:{val_top1:.4f}   | Val Top5:{val_top5:.4f}')
            state_dict = self.student.state_dict()

            if val_acc > best_acc:
                early_stopping = 0
                best_epoch = epoch
                best_loss = val_loss
                best_acc = val_top1
                best_acc2 = val_top5

                torch.save({'epoch' :epoch,
                            'state_dict' :state_dict,
                            'optimizer': self.optimizer_s.state_dict(),
                            'scheduler': self.scheduler_s.state_dict(),
                            }, os.path.join(self.save_path, 'best_model_finetune.pth'))
            else:
                early_stopping += 1

            if early_stopping == args.patience:
                break

            if self.writer is not None:
                self.writer.add_scalar('Best/top1_accuracy', best_acc, epoch)
                self.writer.add_scalar('Best/top5_accuracy', best_acc2, epoch)
                self.writer.add_scalar('Best/loss', best_loss, epoch)

        end = time.time()
        print(f'Best Epoch:{best_epoch} | Loss:{best_loss:.4f} | Top1:{best_acc:.4f} | Top5:{best_acc2:.4f}')
        print(f'Total Training time:{(end - start) / 60:.3f}Minute')

    def finetune_one_epoch(self):
        self.student.train()
        train_loss = 0
        top1 = 0
        top5 = 0

        for images, targets in tqdm(self.train_dl):
            images = torch.tensor(images, device=self.device, dtype=torch.float32)
            targets = torch.tensor(targets, device=self.device, dtype=torch.long)

            self.student.zero_grad(set_to_none=True)

            preds = self.student(images)

            loss = self.criterion(preds, targets)
            loss.backward()

            self.optimizer_s.step()

            t1, t5 = accuracy(preds, targets, (1, 5))
            train_loss += loss.item()
            top1 += t1
            top5 += t5

        top1 /= len(self.train_dl)
        top5 /= len(self.train_dl)
        train_loss /= len(self.train_dl)

        return train_loss, top1, top5 #train_acc

    def validate(self):
        self.student.eval()
        with torch.no_grad():
            val_loss = 0
            top1 = 0
            top5 = 0

            for images, targets in tqdm(self.val_dl):
                images = torch.tensor(images, device=self.device, dtype=torch.float32)
                targets = torch.tensor(targets, device=self.device, dtype=torch.long)

                preds = self.student(images)
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

    trainer = MPL(args)
    trainer.train()
    trainer.finetune()