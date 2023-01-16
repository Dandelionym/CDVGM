# -*- coding:utf-8 -*-

import numpy as np
import torch

# import mxnet as mx
from utils.data_util import get_sample_indices
from tqdm import tqdm

"""
    train, val, test:                   np.ndarray
    stats:                              dict, two keys: mean and std
    train_norm, val_norm, test_norm:    np.ndarray,shape is the same as original

    return  {'mean': mean, 'std': std},
            train,
            val,
            test
"""
def normalization(train, val, test):
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)

    def normalize(x):
        return (x - mean) / std

    train = (train).transpose(0,2,1,3)
    val = (val).transpose(0,2,1,3)
    test =(test).transpose(0,2,1,3)

    return {'mean': mean, 'std': std}, train, val, test


"""
    graph_signal_matrix_filename:                   str, path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours:        int
    num_for_predict:                                int
    points_per_hour:                                int, default 12, depends on data
    merge:                                          boolean, default False, whether to merge training set and validation set to train model
    
    return all_data
"""
def read_and_generate_dataset(graph_signal_matrix_filename, num_of_weeks, num_of_days, num_of_hours, num_for_predict, points_per_hour=12, merge=False):
    data_seq = np.load(graph_signal_matrix_filename)['data']      # (17856, 170, 3)

    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days, num_of_hours, idx, num_for_predict, points_per_hour)
        if not sample:
            continue

        week_sample, day_sample, hour_sample, target = sample
        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
        ))

    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    if not merge:
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line1])]
    else:
        print('Merge training set and validation set!')
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line2])]

    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    train_week, train_day, train_hour, train_target = training_set
    val_week, val_day, val_hour, val_target         = validation_set
    test_week, test_day, test_hour, test_target     = testing_set

    train_hour = torch.unsqueeze(torch.tensor(train_hour[:, :, 0, :]), -2).numpy()
    val_hour = torch.unsqueeze(torch.tensor(val_hour[:, :, 0, :]), -2).numpy()
    test_hour = torch.unsqueeze(torch.tensor(test_hour[:, :, 0, :]), -2).numpy()

    print("                  week_shape  	 	   day_shape        hour_shape[selected]       target_shape")
    print('train data    {}   {}   {}   {}'.format(train_week.shape, train_day.shape, train_hour.shape, train_target.shape))
    print('valid data    {}   {}   {}   {}'.format(val_week.shape, val_day.shape, val_hour.shape, val_target.shape))
    print('tests data    {}   {}   {}   {}'.format(test_week.shape, test_day.shape, test_hour.shape, test_target.shape))


    (week_stats, train_week_norm,
     val_week_norm, test_week_norm) = normalization(train_week,
                                                    val_week,
                                                    test_week)

    (day_stats, train_day_norm,
     val_day_norm, test_day_norm) = normalization(train_day,
                                                  val_day,
                                                  test_day)

    (recent_stats, train_recent_norm,
     val_recent_norm, test_recent_norm) = normalization(train_hour,
                                                        val_hour,
                                                        test_hour)

    all_data = {
        'train': {
            'week': train_week_norm,
            'day': train_day_norm,
            'recent': train_recent_norm,
            'target': train_target,
        },
        'val': {
            'week': val_week_norm,
            'day': val_day_norm,
            'recent': val_recent_norm,
            'target': val_target
        },
        'test': {
            'week': test_week_norm,
            'day': test_day_norm,
            'recent': test_recent_norm,
            'target': test_target
        },
        'stats': {
            'week': week_stats,
            'day': day_stats,
            'recent': recent_stats
        }
    }

    return all_data
