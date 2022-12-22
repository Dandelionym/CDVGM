# -*- coding:utf-8 -*-

import csv
import numpy as np
from scipy.sparse.linalg import eigs
from .metrics import mean_absolute_error, mean_squared_error, masked_mape_np
import torch


def search_data(sequence_length, num_of_batches, label_start_idx,
				num_for_predict, units, points_per_hour):
	'''
	Parameters
	----------
	sequence_length: int, length of all history data
	num_of_batches: int, the number of batches will be used for training
	label_start_idx: int, the first index of predicting target
	num_for_predict: int,
					 the number of points will be predicted for each sample
	units: int, week: 7 * 24, day: 24, recent(hour): 1
	points_per_hour: int, number of points per hour, depends on data
	Returns
	----------
	list[(start_idx, end_idx)]
	'''
	
	if points_per_hour < 0:
		raise ValueError("points_per_hour should be greater than 0!")
	
	if label_start_idx + num_for_predict > sequence_length:
		return None
	
	x_idx = []
	for i in range(1, num_of_batches + 1):
		start_idx = label_start_idx - points_per_hour * units * i
		end_idx = start_idx + num_for_predict
		if start_idx >= 0:
			x_idx.append((start_idx, end_idx))
		else:
			return None
	
	if len(x_idx) != num_of_batches:
		return None
	
	return x_idx[::-1]  # 倒叙输出,符合时间的 顺序输出,这里不占用多少空间


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
					   label_start_idx, num_for_predict, points_per_hour=12):
	'''
	Parameters
	----------
	data_sequence: np.ndarray
				   shape is (sequence_length, num_of_vertices, num_of_features)
	num_of_weeks, num_of_days, num_of_hours: int
	label_start_idx: int, the first index of predicting target
	num_for_predict: int,
					 the number of points will be predicted for each sample
	points_per_hour: int, default 12, number of points per hour
	Returns
	----------
	week_sample: np.ndarray
				 shape is (num_of_weeks * points_per_hour,
						   num_of_vertices, num_of_features)
	day_sample: np.ndarray
				 shape is (num_of_days * points_per_hour,
						   num_of_vertices, num_of_features)
	hour_sample: np.ndarray
				 shape is (num_of_hours * points_per_hour,
						   num_of_vertices, num_of_features)
	target: np.ndarray
			shape is (num_for_predict, num_of_vertices, num_of_features)
	'''
	week_indices = search_data(data_sequence.shape[0], num_of_weeks,
							   label_start_idx, num_for_predict,
							   7 * 24, points_per_hour)
	if not week_indices:
		return None
	
	day_indices = search_data(data_sequence.shape[0], num_of_days,
							  label_start_idx, num_for_predict,
							  24, points_per_hour)
	if not day_indices:
		return None
	
	hour_indices = search_data(data_sequence.shape[0], num_of_hours,
							   label_start_idx, num_for_predict,
							   1, points_per_hour)
	if not hour_indices:
		return None
	
	week_sample = np.concatenate([data_sequence[i: j] for i, j in week_indices], axis=0)
	day_sample = np.concatenate([data_sequence[i: j] for i, j in day_indices], axis=0)
	hour_sample = np.concatenate([data_sequence[i: j] for i, j in hour_indices], axis=0)
	target = data_sequence[label_start_idx: label_start_idx + num_for_predict]
	
	return week_sample, day_sample, hour_sample, target


def get_adjacency_matrix(distance_df_filename, num_of_vertices):
	with open(distance_df_filename, 'r') as f:
		reader = csv.reader(f)
		header = f.__next__()
		edges = [(int(i[0]), int(i[1])) for i in reader]
	A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
				 dtype=np.float32)
	for i, j in edges:
		A[i, j] = 1
	return A


def scaled_Laplacian(W):
	assert W.shape[0] == W.shape[1]
	D = np.diag(np.sum(W, axis=1))
	L = D - W
	lambda_max = eigs(L, k=1, which='LR')[0].real
	
	return (2 * L) / lambda_max - np.identity(W.shape[0])


def compute_val_loss(net, val_loader, loss_function, device, epoch,):
	net.eval()
	with torch.no_grad():
		tmp = []
		for index, (_, _, val_r, val_t) in enumerate(val_loader):
			# val_w = val_w.to(device)
			# val_d = val_d.to(device)
			val_r = val_r.to(device)
			val_t = val_t.to(device)
			output, _, _ = net(val_r)
			loss_ = loss_function(output, val_t)
			tmp.append(loss_.item())
		
		validation_loss = sum(tmp) / len(tmp)
		print('\033[37m [Epoch: %s]   \033[31m V-Loss => %.4f' % (epoch, validation_loss))
		return validation_loss


def predict(net, test_loader, device):
	net.eval()
	with torch.no_grad():
		prediction = []
		for index, (_, _, test_r, _) in enumerate(test_loader):
			test_r = test_r.to(device)
			output, _, _ = net(test_r)
			prediction.append(output.cpu().detach().numpy())
		prediction = np.concatenate(prediction, 0)
		return prediction

def evaluate(net, test_loader, true_value, device):
	net.eval()
	with torch.no_grad():
		prediction = predict(net, test_loader, device)
		mem_mae  = []
		mem_mape = []
		mem_rmse = []
		for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
			print('\033[37m    Pred %s min\t' % (i*5, ), end='')
			mae = mean_absolute_error(true_value[:, :, 0:i], prediction[:, :, 0:i])
			rmse = mean_squared_error(true_value[:, :, 0:i], prediction[:, :, 0:i]) ** 0.5
			mape = masked_mape_np(true_value[:, :, 0:i], prediction[:, :, 0:i], 0)
			print('  MAE: %.5f \t' % (mae, ), 'RMSE: %.5f\t' % (rmse, ), 'MAPE: %.5f' % (mape, ))
			mem_mae.append(mae)
			mem_mape.append(mape)
			mem_rmse.append(rmse)
		_MAE = mean_absolute_error(true_value[:, :, 0:12], prediction[:, :, 0: 12])
		_RMSE = mean_squared_error(true_value[:, :, 0:12], prediction[:, :, 0:12]) ** 0.5
		_MAPE = masked_mape_np(true_value[:, :, 0:12], prediction[:, :, 0:12], 0)
		print(f'\t MAE : {np.array(mem_mae).mean()}    RMSE : {np.array(mem_rmse).mean()}   MAPE : {np.array(mem_mape).mean()}')
	return np.array(mem_mae).mean(), np.array(mem_rmse).mean(), np.array(mem_mape).mean()


