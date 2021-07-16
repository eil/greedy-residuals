#!/bin/bash
# Train ResNet on CIFAR-10 with Batch Norm for Baseline
python classifier.py --norm-type bn --arch resnet --epochs 200
# Train ResNet on CIFAR-10 for Baseline
python classifier.py --norm-type none --arch resnet --epochs 200
# Train embedded ResNet on CIFAR-10 with Batch Norm
python classifier.py --norm-type bn --embed --arch resnet --epochs 200
# Train embedded ResNet on CIFAR-10
python classifier.py --norm-type none --embed --arch resnet --epochs 200
# Train ResNet on CIFAR-100 with Batch Norm for Baseline
python classifier.py --norm-type bn --dataset cifar100 --arch resnet --epochs 200
# Train ResNet on CIFAR-100 for Baseline
python classifier.py --norm-type none --dataset cifar100 --arch resnet --epochs 200
# Train embedded ResNet on CIFAR-100 with Batch Norm
python classifier.py --norm-type bn --embed --dataset cifar100 --arch resnet --epochs 200
# Train embedded ResNet on CIFAR-100
python classifier.py --norm-type none --embed --dataset cifar100 --arch resnet --epochs 200
# Train ResNet on Caltech-101 with Batch Norm for Baseline
python classifier.py --norm-type bn --dataset caltech-101 --arch resnet --epochs 200
# Train ResNet on Caltech-101 for Baseline
python classifier.py --norm-type none --dataset caltech-101 --arch resnet --epochs 200
# Train embedded ResNet on Caltech-101 with Batch Norm
python classifier.py --norm-type bn --embed --dataset caltech-101 --arch resnet --epochs 200
# Train embedded ResNet on Caltech-101
python classifier.py --norm-type none --embed --dataset caltech-101 --arch resnet --epochs 200
# Train ResNet on Caltech-256 with Batch Norm for Baseline
python classifier.py --norm-type bn --dataset caltech-256 --arch resnet --epochs 200
# Train ResNet on Caltech-256 for Baseline
python classifier.py --norm-type none --dataset caltech-256 --arch resnet --epochs 200
# Train embedded ResNet on Caltech-256 with Batch Norm
python classifier.py --norm-type bn --embed --dataset caltech-256 --arch resnet --epochs 200
# Train embedded ResNet on Caltech-256
python classifier.py --norm-type none --embed --dataset caltech-256 --arch resnet --epochs 200