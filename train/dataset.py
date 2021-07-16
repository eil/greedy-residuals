import errno
import os
from collections import defaultdict

import urllib
from urllib.error import URLError

from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader, make_dataset, IMG_EXTENSIONS


class Caltech101(Dataset):
    link = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz'
    filename = '101_ObjectCategories.tar.gz'
    foldername = '101_ObjectCategories'

    @staticmethod
    def _find_classes(folder):
        classes = [d.name for d in os.scandir(folder) if d.is_dir()]
        # classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __init__(self, root, train=True, transform=None, download=True):
        self.root = root
        root = os.path.join(root, self.foldername)

        if download:
            self.download()

        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS)

        datapaths = defaultdict(list)

        for path, target in samples:
            datapaths[target].append(path)

        for target in datapaths.keys():
            if train:
                datapaths[target] = datapaths[target][:int(0.8 * len(datapaths[target]))]
            else:
                datapaths[target] = datapaths[target][int(0.8 * len(datapaths[target])):]

        newdatapaths = []
        labels = []
        for target in datapaths.keys():
            for path in datapaths[target]:
                newdatapaths.append(path)
                labels.append(target)

        self.train = train
        self.transform = transform
        self.labels = labels
        self.datapaths = newdatapaths
        self.cache = {}

    def __getitem__(self, index):
        target = self.labels[index]
        if index in self.cache:
            img = self.cache[index]
        else:
            path = self.datapaths[index]
            img = pil_loader(path)
            self.cache[index] = img

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def download(self):
        import tarfile

        if os.path.exists(os.path.join(self.root, self.filename)):
            print('Files already downloaded and verified')
            return

        root = os.path.expanduser(self.root)
        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # downloads file
        try:
            print('Downloading ' + self.link + ' to ' + fpath)
            urllib.request.urlretrieve(self.link, fpath)
        except URLError:
            if self.link[:5] == 'https':
                url = self.link.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)

        # extract file
        cwd = os.getcwd()
        mode = 'r:gz' if self.filename.endswith('.gz') else 'r'
        tar = tarfile.open(os.path.join(self.root, self.filename), mode)
        os.chdir(self.root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __len__(self):
        return len(self.datapaths)


class Caltech256(Caltech101):
    link = 'http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar'
    filename = '256_ObjectCategories.tar'
    foldername = '256_ObjectCategories'


class ImageNet(Caltech101):
    # Supposed to be download manually
    link = 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar'
    filename = 'ILSVRC2012_img_train.tar'
    foldername = 'ILSVRC2012'




