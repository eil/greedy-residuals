# Watermarking Deep Neural Networks with Greedy Residuals
The official implementation codes of greedy residuals for the paper [Watermarking Deep Neural Networks with Greedy Residuals](http://proceedings.mlr.press/v139/liu21x.html) (__ICML 2021__).

In this work, we propose a novel DNN watermarking method called *greedy residuals*. Our essential insight is that by making greedy residuals depend on less information, a more robust watermarking method can be constructed (i.e., *less is more*). There are two main aspects of the understanding of *less* here: 
1. we greedily select those fewer and more important model parameters for embedding, and the residuals are built upon the selected parameters; 
2. we hardly need to use external data sources, that is, we do not need explicit ownership indicators to complete ownership verification, since ownership information in the residuals can be verified with only fixed and simple steps.

## Prerequisites

### Dataset
The dataset will be downloaded automatically. 
Also, you can manually download the [CIFAR-10 & CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), [Caltech-101 & Caltech-256](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) and [ImageNet](http://image-net.org/download) dataset and move them to the folder `data/`. 
The folder structure should look like:
```
.
|-- caltech-101
|   `-- 101_ObjectCategories
|-- caltech-256
|   `-- 256_ObjectCategories
|-- cifar10
|   `-- cifar-10-batches-py
|-- cifar100
|   `-- cifar-100-python
`-- imagenet
    |-- ILSVRC2012
    `-- val
```

Note that for ImageNet, in the experiments we only use the first 100-class subset as the entire dataset for efficiency.

### Dependencies 
You can install the tested dependencies for this repository by using
```bash
pip install rsa torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

## Evaluations
### Training
Taking ResNet-18 as an example, to train the baseline models and watermarked models, simply run the following:
```commandline
bash ./scripts/train.sh
```
This `train.sh` script will automatically train ResNet-18 on CIFAR-10, CIFAR-100, Caltech-101 and Caltech-256 respectively. 
More details can be found in the script file.
The results are saved in the folder `logs/`.

### Fine-tuning
To fine-tune the baseline models and watermarked models between CIFAR-10, CIFAR-100, Caltech-101 and Caltech-256 for ResNet-18, run the following script:
```commandline
bash ./scripts/finetune.sh
```
You can see the script file for more details. Note that you should run the `train.sh` script first and then run the `finetune.sh` script.

### Other usages
Please check the paper and source codes for more details.

## Citation
If you find this work useful for your research, please cite
```
@inproceedings{pmlr-v139-liu21x,
  title = 	 {Watermarking Deep Neural Networks with Greedy Residuals},
  author =       {Liu, Hanwen and Weng, Zhenyu and Zhu, Yuesheng},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {6978--6988},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/liu21x/liu21x.pdf},
  url = 	 {http://proceedings.mlr.press/v139/liu21x.html},
}
```
