from typing import List
import math

import torch
import torch.nn as nn
import torch.nn.init as init


class AvgPooling(nn.Module):
    def __init__(self, input_len, output_len):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len

        def kernels(ind, outd) -> List:

            def start_index(a, b, c):
                return math.floor((float(a - 1) * float(c)) / b)

            def end_index(a, b, c):
                return math.ceil((float(a) * float(c)) / b)

            results = []
            for ow in range(1, outd + 1):
                start = start_index(ow, outd, ind) + 1  # math.floor(((i - 1) * m) / d)
                end = end_index(ow, outd, ind) + 1  # math.ceil((i * m) / d)
                sz = end - start  # \sigma = math.ceil((i * m) / d) - math.floor(((i - 1) * m) / d)
                results.append((start, sz))
            return results

        self.kernels = kernels(input_len, output_len)
        self.kernels_len = len(self.kernels)

    def forward(self, x):
        y = torch.zeros(self.output_len)
        for idx, k in enumerate(self.kernels):
            y[idx] = (x[k[0] - 1: k[0] + k[1] - 1] / k[1]).sum()
        return y


class ConvBlock(nn.Module):
    def __init__(self, i, o, ks=3, s=1, pd=1, bn='bn', relu=True):
        super().__init__()

        self.conv = nn.Conv2d(i, o, ks, s, pd, bias=(bn == 'none'))

        self.bn = nn.BatchNorm2d(o) if bn == 'bn' else None

        self.relu = nn.ReLU(inplace=True) if relu else None

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
