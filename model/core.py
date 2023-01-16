import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, InstanceNorm2d
from torch.nn.utils import weight_norm

from model.utils import ST_BLOCK_1


class CDVGM(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, week, day, recent):
        super(CDVGM, self).__init__()
        tem_size = recent
        self.num_nodes = num_nodes
        self.bn = BatchNorm2d(c_in, affine=False)
        self.block1 = ST_BLOCK_1(c_in, c_out, num_nodes, tem_size)
        self.block2 = ST_BLOCK_1(c_out, c_out, num_nodes, tem_size)
        self.block3 = ST_BLOCK_1(c_out, c_out, num_nodes, tem_size)
        self.block4 = ST_BLOCK_1(c_out, c_out, num_nodes, tem_size)
        self.norm = BatchNorm2d(c_in, num_nodes, tem_size, affine=False)
        self.down_ = Conv2d(c_in, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=False)

        self.conv1 = Conv2d(c_out, c_out // 2, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=False)
        self.conv2 = Conv2d(c_out // 2, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=False)
        self.conv3 = Conv2d(c_out, c_out // 2, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=False)
        self.conv4 = Conv2d(c_out // 2, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=False)

        self.TCN = TemporalConvNet(num_inputs=num_nodes, num_channels=[170, 170])

        self.h = Parameter(torch.rand(num_nodes, num_nodes), requires_grad=True)
        self.b = Parameter(torch.rand(num_nodes, num_nodes), requires_grad=True)
        nn.init.kaiming_normal_(self.h, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.b, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x_r):
        x = torch.cat((x_r,), -1)
        compress_x = torch.sigmoid(torch.squeeze(self.down_(x), 1))
        HPQ = torch.matmul(self.h, self.h.T) + self.b - torch.matmul(compress_x, torch.log(
            1 + compress_x.permute(0, 2, 1)))  # B,N,N
        HPQ = torch.sigmoid(HPQ)
        T = HPQ.type(torch.float32).to(torch.device('cuda:0'))
        Trend = F.leaky_relu(sum(T, 0) * 0.00001)
        A1 = F.dropout(Trend, 0.5, self.training)
        A1 = torch.mean(A1) - torch.matmul(A1, A1.T)
        x, d_adj, d_adj = self.block1(x, A1)
        A1 = torch.mean(A1) - torch.matmul(A1, A1.T)
        x, d_adj, t_adj = self.block2(x, A1)
        A1 = torch.mean(A1) - torch.matmul(A1, A1.T)
        x, d_adj, t_adj = self.block3(x, A1)
        x, d_adj, t_adj = self.block4(x, A1)

        x1 = x[:, :, :, 0:12]
        x2 = x[:, :, :, 12:24]

        x1 = self.conv1(x1).squeeze()
        x1 = self.conv2(x1).squeeze()
        x2 = self.conv3(x2).squeeze()
        x2 = self.conv4(x2).squeeze()

        x = x1 + x2
        x = self.TCN(x)

        return x, d_adj, A1


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
