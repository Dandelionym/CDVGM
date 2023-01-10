import os
import numpy as np
import torch
from tqdm import tqdm
from time import time
import torch.nn as nn
from preprocess import *
import torch.optim as optim
from model.optimize import Lookahead
from model.core import CDVGM as Framework
from data.lib.utils import compute_val_loss, evaluate, predict
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn import BatchNorm2d

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if __name__ == "__main__":
    loss_function = nn.MSELoss()
    net = Framework(c_in=num_of_features, c_out=64, num_nodes=num_nodes, week=24, day=12, recent=24)
    print("\n\nActiveGCN has {} parameters in total.\n\n".format(sum(x.numel() for x in net.parameters())))
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=wdecay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)
    optimizer = Lookahead(optimizer=optimizer)
    net.to(device)

    start_time_all = time()
    his_loss, v_loss_lst, train_time, total_time = [], [], [], time()
    with open(f'./root/record.csv', mode='w', encoding='utf-8') as f:
        f.write(f"seed,epoch,train_loss,valid_loss,learning_rate,_MAE,_MAPE,_RMSE,datetime\n")

    for epoch in range(1, epochs + 1):
        train_loss = []
        start_time_train = time()
        for _, _, train_r, train_t in tqdm(train_loader, ncols=60, smoothing=0.9, colour='blue'):
            train_r = train_r.to(device)
            train_t = train_t.to(device)
            net.train()
            optimizer.zero_grad()
            output, _, A1 = net(train_r)
            loss = loss_function(output, train_t)
            loss.backward()
            optimizer.step()
            training_loss = loss.item()
            train_loss.append(training_loss)

        start_time_all = time()
        scheduler.step()
        end_time_train = time()
        train_loss = np.mean(train_loss)
        print('\033[37m [Epoch: %s]   \033[31m T-Loss => %.4f' % (epoch, np.mean(train_loss)))
        train_time.append(end_time_train - start_time_train)
        valid_loss = compute_val_loss(net, val_loader, loss_function, device, epoch)
        his_loss.append(valid_loss)
        _MAE, _RMSE, _MAPE = evaluate(net, test_loader, true_value, device)

        # Epoch Record
        with open(f'./root/record.csv', mode='a', encoding='utf-8') as f:
            f.write(
                f"{seed},{epoch},{train_loss},{valid_loss},{scheduler.get_last_lr()[0]},{_MAE},{_MAPE},{_RMSE},{datetime.now()}\n")

        params_filename = os.path.join(params_path,
                                       '%s_epoch_%s_%s.params' % (model_name, epoch, str(round(valid_loss, 2))))
        torch.save(net.state_dict(), params_filename)

        v_loss_lst.append(float(valid_loss))
        watch_early_stop = np.array(v_loss_lst)
        arg = np.argmin(watch_early_stop)
        data__ = pd.read_csv(f'./root/record.csv')
        _MAE_ = data__['_MAE']
        _RMSE_ = data__['_RMSE']
        _MAPE_ = data__['_MAPE']
        print(f"\033[36m\n\tLeaderboard：     {min(_MAE_):.6}     {min(_RMSE_):.6}     {min(_MAPE_):.6}\n")
        print(f"\033[36m\tCurrent Best [{arg + 1} / {epoch}]  V-loss  =>  {v_loss_lst[arg]}\n\n")

    print("\033[32m\n\nHoooo! Training finished.\n")
    record = pd.read_csv(f'./root/record.csv')
    _MAE = record['_MAE']
    _RMSE = record['_RMSE']
    _MAPE = record['_MAPE']
    min_MAE = min(_MAE)
    min_RMSE = min(_RMSE)
    min_MAPE = min(_MAPE)
    print(f"Final Result: =>  \n\tMAE  {min_MAE}\t\tRMSE  {min_RMSE}\t\tMAPE  {min_MAPE}")

    print("Training time/epoch: %.4f secs/epoch" % np.mean(train_time))
    bestId = np.argmin(his_loss)
    print("The valid loss on best model is epoch%s, value is %s" % (str(bestId + 1), str(round(his_loss[bestId], 4))))
    best_params_filename = os.path.join(params_path, '%s_epoch_%s_%s.params' % (
        model_name, str(bestId + 1), str(round(his_loss[bestId], 2))))
    net.load_state_dict(torch.load(best_params_filename))
    prediction = predict(net, test_loader, device)

    evaluate(net, test_loader, true_value, device)
    print("Total time: %f s" % (datetime.now() - timestamp_s).seconds)
