import csv
import numpy as np
from scipy.sparse.linalg import eigs
import torch


def search_data(sequence_length, num_of_batches, label_start_idx,
                num_for_predict, units, points_per_hour):
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

    week_sample = np.concatenate([data_sequence[i: j]
                                  for i, j in week_indices], axis=0)
    day_sample = np.concatenate([data_sequence[i: j]
                                 for i, j in day_indices], axis=0)
    hour_sample = np.concatenate([data_sequence[i: j]
                                  for i, j in hour_indices], axis=0)
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1]), float(i[2])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i, j, k in edges:
        A[i, j] = k

    return A


def scaled_Laplacian(W):
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def compute_val_loss(net, val_loader, loss_function, supports, device, epoch):
    net.eval()
    with torch.no_grad():
        tmp = []
        for index, (val_w, val_d, val_r, val_t) in enumerate(val_loader):
            val_w = val_w.to(device)
            val_d = val_d.to(device)
            val_r = val_r.to(device)
            val_t = val_t.to(device)
            output, _, _ = net(val_w, val_d, val_r, supports)
            l = loss_function(output, val_t)
            tmp.append(l.item())
        validation_loss = sum(tmp) / len(tmp)
        print('epoch: %s, validation loss: %.2f' % (epoch, validation_loss))
        return validation_loss


def predict(net, test_loader, supports, device):
    net.eval()
    with torch.no_grad():
        prediction = []
        for index, (test_w, test_d, test_r, test_t) in enumerate(test_loader):
            test_w = test_w.to(device)
            test_d = test_d.to(device)
            test_r = test_r.to(device)
            test_t = test_t.to(device)
            output, _, _ = net(test_w, test_d, test_r, supports)
            prediction.append(output.cpu().detach().numpy())

        # get first batch's spatial attention matrix
        for index, (test_w, test_d, test_r, test_t) in enumerate(test_loader):
            test_w = test_w.to(device)
            test_d = test_d.to(device)
            test_r = test_r.to(device)
            test_t = test_t.to(device)
            _, spatial_at, temporal_at = net(test_w, test_d, test_r, supports)
            spatial_at = spatial_at.cpu().detach().numpy()
            temporal_at = temporal_at.cpu().detach().numpy()
            break

        prediction = np.concatenate(prediction, 0)
        return prediction, spatial_at, temporal_at


def evaluate(net, test_loader, true_value, supports, device, epoch):
    net.eval()
    with torch.no_grad():
        prediction, _, _ = predict(net, test_loader, supports, device)
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(true_value[:, :, 0:i], prediction[:, :, 0:i])
            rmse = mean_squared_error(true_value[:, :, 0:i], prediction[:, :, 0:i]) ** 0.5
            mape = masked_mape_np(true_value[:, :, 0:i], prediction[:, :, 0:i], 0)

        print('MAE: %.2f' % (mae))
        print('RMSE: %.2f' % (rmse))
        print('MAPE: %.2f' % (mape))


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                                y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100
