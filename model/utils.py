import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, BatchNorm1d


class T_cheby_conv(nn.Module):

	def __init__(self, c_in, c_out):
		super(T_cheby_conv, self).__init__()
		c_in_new = 4 * c_in
		self.K = 4
		self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, 3), padding=(0, 1), stride=(1, 1), bias=True)

	def forward(self, x, adj):
		nSample, feat_in, nNode, length = x.shape
		Ls = []
		L1 = adj
		L0 = torch.eye(nNode).cuda()
		Ls.append(L0)
		Ls.append(L1)

		for k in range(2, self.K):
			L2 = 2 * torch.matmul(adj, L1) - L0
			L0, L1 = L1, L2
			Ls.append(L2)

		Lap = torch.stack(Ls, 0)
		Lap = Lap.transpose(-1, -2)

		x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
		x = x.view(nSample, -1, nNode, length)
		out = self.conv1(x)
		return out


class ST_BLOCK_1(nn.Module):
	def __init__(self, c_in, c_out, num_nodes, tem_size):
		super(ST_BLOCK_1, self).__init__()
		self.c_out = c_out

		self.c_channel = Conv2d(c_in, 64, kernel_size=(1, 1), stride=(1, 1), bias=True)
		self.n_channel = Conv2d(c_in,  1, kernel_size=(1, 1), stride=(1, 1), bias=True)
		self.d_channel_a = Conv2d(c_out,      c_out, kernel_size=(1, 1), stride=(1, 1), bias=True)
		self.d_channel_d = Conv2d(c_out // 2,      c_out, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.time   = Conv1d(num_nodes, num_nodes, 3, bias=True)
		self.time_c = Conv2d(1, c_out, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.t_attention = T_Attention(c_out, num_nodes, tem_size)
		self.dynamic_gcn = T_cheby_conv(c_out, 2 * c_out)
		self.output1 = Conv2d(c_out, c_out, kernel_size=(1, 1), stride=(1, 1), bias=True)
		self.output2 = Conv2d(c_out, c_out, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.bn = LayerNorm([c_out, num_nodes, tem_size])

	def forward(self, x, A1):
		x_res = self.c_channel(x)

		x_s = F.dropout(x_res, 0.5, self.training)
		x_g = self.dynamic_gcn(x_s, A1)
		hole, pos = torch.split(x_g, [self.c_out, self.c_out], 1)
		x_g = torch.sigmoid(pos) * F.leaky_relu(hole)

		x = F.dropout(x, 0.5, self.training)
		x_0 = self.n_channel(x)			# Down Sample to 1 channel
		time = torch.squeeze(x_0, 1).contiguous()
		time_conv = torch.sigmoid(self.time(time))
		time_conv = torch.cat((
			torch.unsqueeze(time[:, :, 0], -1),
			time_conv,
			torch.unsqueeze(time[:, :, -1], -1)
		), -1)											# B, N, L

		time_conv = self.TCN(time_conv)
		time_conv = torch.unsqueeze(time_conv, 1)
		x_1 = self.time_c(time_conv).contiguous()
		x_2 = x_res + x_1 - torch.unsqueeze(torch.mean(x_1, -1), -1)
		T_coef = self.t_attention(x_2)
		T_coef = T_coef.transpose(-1, -2)
		x_g_t = torch.einsum('bcnl,blq->bcnq', x_g, T_coef)

		x_g_t = self.output1(x_g_t)
		x_g_t = self.output2(x_g_t)
		x_s = F.leaky_relu(x_g_t)
		out = self.bn(x_res + x_s)
		return out, A1, T_coef


class T_Attention(nn.Module):
	def __init__(self, c_in, num_nodes, tem_size):
		super(T_Attention, self).__init__()
		self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)

		self.conv_d1 = Conv1d(tem_size, tem_size, kernel_size=(2,), dilation=(2,), padding=(1,), bias=False)
		self.conv_d2 = Conv1d(c_in,         c_in, kernel_size=(2,), dilation=(2,), padding=(1,), bias=False)
		self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
		nn.init.xavier_uniform_(self.w)
		self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)
		self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
		nn.init.xavier_uniform_(self.v)
		self.bn = BatchNorm1d(tem_size)

	def forward(self, seq):
		c1 = seq.permute(0, 1, 3, 2)
		f1 = self.conv1(c1).squeeze()
		f1 = self.conv_d1(f1)
		c2 = seq.permute(0, 2, 1, 3)
		f2 = self.conv2(c2).squeeze()
		f2 = self.conv_d2(f2)
		temp1 = torch.matmul(f1, self.w)
		temp2 = torch.matmul(temp1, f2)
		logits = torch.sigmoid(temp2 + self.b)
		logits = torch.matmul(self.v, logits)
		logits = logits.permute(0, 2, 1).contiguous()
		logits = self.bn(logits).permute(0, 2, 1).contiguous()
		coef_ = torch.softmax(logits, -1)
		return coef_
