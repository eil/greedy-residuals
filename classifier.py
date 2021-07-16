import argparse
from pprint import pprint

from train.classification import Classification


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--action', default='train', choices=['train', 'eval', 'fine-tune'],
                        help='classification experiments (default: train)')

    parser.add_argument('--arch', default='resnet', choices=['resnet'],
                        help='architecture (default: resnet)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size (default: 64)')
    parser.add_argument('--epochs', type=int, required=True,
                        help='experiment epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10',
                                                                 'cifar100',
                                                                 'caltech-101',
                                                                 'caltech-256',
                                                                 'imagenet'],
                        help='experiment dataset (default: cifar10)')
    parser.add_argument('--norm-type', default='bn', choices=['bn', 'none'],
                        help='norm type (default: bn)')

    # watermark
    parser.add_argument('--embed', action='store_true', default=False,
                        help='turn on watermarking')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='threshold for watermarking (default: 0.1)')
    parser.add_argument('--lamda', type=float, default=0.01,
                        help='coe of watermark reg in loss function (default: 0.01)')
    parser.add_argument('--divider', type=int, default=2,
                        help='describe the fraction of elements to be zeros in watermarking (default: 2)')

    # paths
    parser.add_argument('--pretrained-path',
                        help='path of pretrained model')
    parser.add_argument('--lr-config', default='default.json',
                        help='lr config json file')

    args = parser.parse_args()

    pprint(vars(args))

    classification = Classification(vars(args))
    if classification.action == 'train':
        classification.training()
    elif classification.action == 'eval':
        classification.evaluate()
    elif classification.action == 'fine-tune':
        classification.training(tuning=True)

    print('The logs can be found at', classification.logdir)
