import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conv2d import ConvBlock
from models.conv2d import AvgPooling


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_type='bn'):
        super(BasicBlock, self).__init__()

        self.convbnrelu_1 = ConvBlock(in_planes, planes, 3, stride, 1, bn=norm_type, relu=True)
        self.convbn_2 = ConvBlock(planes, planes, 3, 1, 1, bn=norm_type, relu=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = ConvBlock(in_planes, self.expansion * planes,
                                      1, stride, 0, bn=norm_type, relu=False)

    def forward(self, x):
        out = self.convbnrelu_1(x)
        out = self.convbn_2(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_type='bn'):
        super(Bottleneck, self).__init__()

        self.convbnrelu_1 = ConvBlock(in_planes, planes, 1, 1, 0, bn=norm_type, relu=True)
        self.convbnrelu_2 = ConvBlock(planes, planes, 3, stride, 1, bn=norm_type, relu=True)
        self.convbn_3 = ConvBlock(planes, self.expansion * planes, 1, 1, 0, bn=norm_type, relu=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = ConvBlock(in_planes, self.expansion * planes, 1, stride, 0, bn=norm_type, relu=False)

    def forward(self, x):
        out = self.convbnrelu_1(x)
        out = self.convbnrelu_2(out)
        out = self.convbn_3(out) + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_type='bn'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks
        self.norm_type = norm_type

        self.convbnrelu_1 = ConvBlock(3, 64, 3, 1, 1, bn=norm_type, relu=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm_type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.convbnrelu_1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out


def ResNet18(**model_kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **model_kwargs)


class ModelTest(object):
    def __init__(self, model, m_layers, objective_size):
        self.model = model
        self.m_layers = m_layers
        self.objective_size = objective_size

    def extract_weight(self):
        extraction = None
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if 'weight' in name:
                if name in self.m_layers:
                    layer_weight = param.view(-1)[
                                   :param.numel() // self.objective_size[1] * self.objective_size[1]]
                    m = len(layer_weight)
                    print(layer_weight)

                    in_length = m
                    out_length = self.objective_size[0] * self.objective_size[1]
                    print(f'm={in_length}, d={out_length}')
                    my_conv = AvgPooling(in_length, out_length)

                    conved = my_conv(layer_weight.view(-1)).view(self.objective_size)

                    layer_weight = nn.functional.adaptive_avg_pool1d(layer_weight[None, None],
                                                                     self.objective_size[0] * self.objective_size[
                                                                         1]).squeeze(
                        0).view(self.objective_size)
                    print(f'Torch pooling = {layer_weight}')
                    print(f'My own pooling = {conved}')
                    print(f'ele-wise cmp = {conved != layer_weight}')
                    print(f'count of the different ele = {torch.sum(conved != layer_weight)}')
                    print(f'count of the different ele under precisions = {torch.sum(conved - layer_weight > 1e-8)}')
                    if extraction is None:
                        extraction = layer_weight
                    else:
                        extraction += layer_weight

        extraction /= len(self.m_layers)
        return extraction


if __name__ == '__main__':
    key_model = ResNet18()
    tester = ModelTest(model=ResNet18(), m_layers=["convbnrelu_1.conv.weight"], objective_size=[256, 253])
    tester.extract_weight()
