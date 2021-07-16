import os
from pprint import pprint

import torch
import torch.optim as optim
from torchvision.transforms import transforms
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torch.utils.data import DataLoader

from models.residual import Residual
from train.dataset import Caltech101, Caltech256, ImageNet
from train.logdb import LogDB
from train.trainer import Trainer
from models.resnet import ResNet18


class Classification(LogDB):
    def __init__(self, args):
        super().__init__(args)
        self.prepare_dataset()

        self.create_folder()

        self.construct_model()
        self.construct_res()

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)

        if len(self.lr_config[self.lr_config['type']]) != 0:  # if no specify steps, then scheduler = None
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       self.lr_config[self.lr_config['type']], self.lr_config['gamma'])
        else:
            scheduler = None

        self.trainer = Trainer(optimizer, scheduler, self.device)

    def construct_model(self):
        print('Loading arch: ' + self.arch)

        if self.arch == 'resnet':
            model = ResNet18(num_classes=self.num_classes, norm_type=self.norm_type)
        else:
            raise Exception('Unknown arch')

        if self.pretrained_path is not None:
            sd = torch.load(self.pretrained_path)
            if self.action == 'fine-tune':
                print('Changing classifier for fine-tuning')
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in sd.items() if k.find('classifier') == -1}
                # update
                model_dict.update(pretrained_dict)
                sd = model_dict
            model.load_state_dict(sd)
        self.model = model.to(self.device)
        pprint(self.model)

    def construct_res(self, parameterization=None, assertion=None, tmp=False):
        if parameterization is None:
            parameterization = {}
        if self.embed:
            residual = Residual(m_layers=self.res_config['m_layers'],  # ["convbnrelu_1.conv.weight"],
                                objective_size=self.res_config['objective_size'],  # [256, 253],
                                threshold=parameterization.get("threshold", self.threshold),
                                lamda=parameterization.get("lamda", self.lamda),
                                divider=parameterization.get("divider", self.divider),
                                assertion=assertion)
            if self.pretrained_path is None:
                self.save_model('best.pth.res', model=residual)
            else:
                if assertion is None:
                    residual.load_state_dict(torch.load(f'{self.pretrained_path}.res'))
            if tmp:
                return residual.to(self.device)
            self.res = residual.to(self.device)

    def training(self, tuning=False):
        best_acc = float('-inf')
        history_file = os.path.join(self.logdir, 'tuning-history.csv' if tuning else 'history.csv')
        first = True

        for ep in range(1, self.epochs + 1):
            train_metrics = self.trainer.train(self.train_loader, self.model, None if tuning else self.res)
            valid_metrics = self.trainer.test(self.test_loader, self.model, self.res)

            metrics = {}
            for key in train_metrics:
                metrics[f'train_{key}'] = train_metrics[key]
            for key in valid_metrics:
                metrics[f'valid_{key}'] = valid_metrics[key]

            self.append_history(history_file, metrics, first)
            first = False

            if best_acc < metrics['valid_acc']:
                best_acc = metrics['valid_acc']
                self.save_model('best.pth')
            for key, value in metrics.items():
                print(f'{key}: {value:6.4f}', end=', ')
            print(f'Best Acc: {best_acc:6.4f}, Epoch: {ep}/{self.epochs}', end='\r')

    def evaluate(self):
        valid_metrics = self.trainer.test(self.test_loader, self.model, self.res)
        for key, value in valid_metrics.items():
            print(f'{key}: {value:6.4f}', end=', ')

    def prepare_dataset(self):
        ds = self.dataset

        is_cifar = 'cifar' in ds
        root = f'data/{ds}'
        print('Loading dataset: ' + ds)

        selected_dataset = {
            'cifar10': CIFAR10,
            'cifar100': CIFAR100,
            'caltech-101': Caltech101,
            'caltech-256': Caltech256,
            'imagenet': ImageNet
        }[ds]

        self.num_classes = {
            'cifar10': 10,
            'cifar100': 100,
            'caltech-101': 102,
            'caltech-256': 257,
            'imagenet': 100,
        }[ds]

        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

        # train transform
        if not is_cifar:
            transform_list = [
                transforms.Resize(32),
                transforms.CenterCrop(32)
            ]
        else:
            transform_list = []

        transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_transforms = transforms.Compose(transform_list)

        # test transform
        if not is_cifar:
            transform_list = [
                transforms.Resize(32),
                transforms.CenterCrop(32)
            ]
        else:
            transform_list = []

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        test_transforms = transforms.Compose(transform_list)

        # dataset and loader
        train_dataset = selected_dataset(root,
                                         train=True,
                                         transform=train_transforms,
                                         download=True)
        test_dataset = selected_dataset(root,
                                        train=False,
                                        transform=test_transforms)
        loader_worker = 4
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=loader_worker,
                                  drop_last=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size * 2,
                                 shuffle=False,
                                 num_workers=loader_worker)
        self.train_loader = train_loader
        self.test_loader = test_loader
