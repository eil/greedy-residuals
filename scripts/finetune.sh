#!/bin/bash
# Fine-tune ResNet from CIFAR-10 to CIFAR-100 with Batch Norm
python classifier.py --action fine-tune --pretrained-path logs/resnet_cifar10_train/1/models/best.pth --epochs 100 --dataset cifar100 --lr 0.001 --norm-type bn --arch resnet
# Fine-tune watermarked ResNet from CIFAR-10 to CIFAR-100 with Batch Norm
python classifier.py --action fine-tune --embed --pretrained-path logs/resnet_cifar10_train_embed/1/models/best.pth --epochs 100 --dataset cifar100 --lr 0.001 --norm-type bn --arch resnet
# Fine-tune ResNet from CIFAR-10 to Caltech-101 with Batch Norm
python classifier.py --action fine-tune --pretrained-path logs/resnet_cifar10_train/1/models/best.pth --epochs 100 --dataset caltech-101 --lr 0.001 --norm-type bn --arch resnet
# Fine-tune watermarked ResNet from CIFAR-10 to Caltech-101 with Batch Norm
python classifier.py --action fine-tune --embed --pretrained-path logs/resnet_cifar10_train_embed/1/models/best.pth --epochs 100 --dataset caltech-101 --lr 0.001 --norm-type bn --arch resnet
# Fine-tune ResNet from CIFAR-10 to Caltech-256 with Batch Norm
python classifier.py --action fine-tune --pretrained-path logs/resnet_cifar10_train/1/models/best.pth --epochs 100 --dataset caltech-256 --lr 0.001 --norm-type bn --arch resnet
# Fine-tune watermarked ResNet from CIFAR-10 to Caltech-256 with Batch Norm
python classifier.py --action fine-tune --embed --pretrained-path logs/resnet_cifar10_train_embed/1/models/best.pth --epochs 100 --dataset caltech-256 --lr 0.001 --norm-type bn --arch resnet
# Fine-tune ResNet from CIFAR-100 to CIFAR-10 with Batch Norm
python classifier.py --action fine-tune --pretrained-path logs/resnet_cifar100_train/1/models/best.pth --epochs 100 --dataset cifar10 --lr 0.001 --norm-type bn --arch resnet
# Fine-tune watermarked ResNet from CIFAR-100 to CIFAR-10 with Batch Norm
python classifier.py --action fine-tune --embed --pretrained-path logs/resnet_cifar100_train_embed/1/models/best.pth --epochs 100 --dataset cifar10 --lr 0.001 --norm-type bn --arch resnet
# Fine-tune ResNet from CIFAR-100 to Caltech-101 with Batch Norm
python classifier.py --action fine-tune --pretrained-path logs/resnet_cifar100_train/1/models/best.pth --epochs 100 --dataset caltech-101 --lr 0.001 --norm-type bn --arch resnet
# Fine-tune watermarked ResNet from CIFAR-100 to Caltech-101 with Batch Norm
python classifier.py --action fine-tune --embed --pretrained-path logs/resnet_cifar100_train_embed/1/models/best.pth --epochs 100 --dataset caltech-101 --lr 0.001 --norm-type bn --arch resnet
# Fine-tune ResNet from CIFAR-100 to Caltech-256 with Batch Norm
python classifier.py --action fine-tune --pretrained-path logs/resnet_cifar100_train/1/models/best.pth --epochs 100 --dataset caltech-256 --lr 0.001 --norm-type bn --arch resnet
# Fine-tune watermarked ResNet from CIFAR-100 to Caltech-256 with Batch Norm
python classifier.py --action fine-tune --embed --pretrained-path logs/resnet_cifar100_train_embed/1/models/best.pth --epochs 100 --dataset caltech-256 --lr 0.001 --norm-type bn --arch resnet