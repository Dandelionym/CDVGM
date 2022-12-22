import os
import shutil
from datetime import datetime
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from data.lib.preprocess import read_and_generate_dataset
from data.lib.utils import scaled_Laplacian, get_adjacency_matrix


np.seterr(divide='ignore', invalid='ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--max_epoch', type=int, default=40, help='Epoch to run [default: 40]')
parser.add_argument('--momentum', type=float, default=0.99, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adamW', help='adam or momentum [default: adam]')
parser.add_argument('--length', type=int, default=24, help='Size of temporal : 12')
parser.add_argument("--force", type=str, default=True, help="remove params dir")
parser.add_argument("--data_name", type=str, default=8, help="the number of data documents [8/4]", required=False)
parser.add_argument('--num_point', type=int, default=170, help='road Point Number [170/307] ', required=False)
parser.add_argument('--decay', type=float, default=0.97, help='decay rate of learning rate [0.97/0.92]')

FLAGS = parser.parse_args()
decay = FLAGS.decay
dataname = FLAGS.data_name
adj_filename = 'data/PEMS0%s/distance.csv' % dataname
graph_signal_matrix_filename = 'data/PEMS0%s/pems0%s.npz' % (dataname, dataname)
Length = FLAGS.length
num_nodes = FLAGS.num_point
epochs = FLAGS.max_epoch
optimizer = FLAGS.optimizer
points_per_hour = 12
num_for_predict = 12
num_of_weeks = 2
num_of_days = 1
num_of_hours = 2
num_of_vertices = FLAGS.num_point
num_of_features = 1
merge = False
model_name = 'ActivateGCN_SOTA_%s' % dataname
params_dir = 'experiment_D_R_another'
prediction_path = 'DGCN_Res_prediction_0%s' % dataname
device = torch.device(FLAGS.device)
wdecay = 0.001
learning_rate = 0.0015
batch_size = 16


timestamp_s = datetime.now()
if params_dir != "None":
    params_path = os.path.join(params_dir, model_name)
else:
    params_path = 'params/%s_%s/' % (model_name, timestamp_s)

if os.path.exists(params_path) and not FLAGS.force:
    raise SystemExit("Params folder exists! Select a new params path please!")
else:
    if os.path.exists(params_path):
        shutil.rmtree(params_path)
    os.makedirs(params_path)
    try:
        os.makedirs("./root")
    except:
        pass
    print('\033[37mCreate params directory %s, reading data...' % (params_path, ))


def generate_all_data(batch_size_):
    all_data = read_and_generate_dataset(graph_signal_matrix_filename,
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         points_per_hour,
                                         merge)

    # test set ground truth
    true_value = all_data['test']['target']

    # training set data loader
    train_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['train']['week']),
            torch.Tensor(all_data['train']['day']),
            torch.Tensor(all_data['train']['recent']),
            torch.Tensor(all_data['train']['target'])
        ),
        batch_size=batch_size_,
        shuffle=True
    )
    # validation set data loader
    val_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['val']['week']),
            torch.Tensor(all_data['val']['day']),
            torch.Tensor(all_data['val']['recent']),
            torch.Tensor(all_data['val']['target'])
        ),
        batch_size=batch_size_,
        shuffle=False
    )

    # testing set data loader
    test_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['test']['week']),
            torch.Tensor(all_data['test']['day']),
            torch.Tensor(all_data['test']['recent']),
            torch.Tensor(all_data['test']['target'])
        ),
        batch_size=batch_size_,
        shuffle=False
    )
    return all_data, true_value, train_loader, val_loader, test_loader


all_data, true_value, train_loader, val_loader, test_loader = generate_all_data(batch_size)
stats_data = {}
for type_ in ['week', 'day', 'recent']:
    stats = all_data['stats'][type_]
    stats_data[type_ + '_mean'] = stats['mean']
    stats_data[type_ + '_std'] = stats['std']
np.savez_compressed(
    os.path.join(params_path, 'stats_data'),
    **stats_data
)