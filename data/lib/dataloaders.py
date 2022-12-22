import torch
from torch.utils.data import DataLoader, TensorDataset


def data_generator(all_data, batch_size):
    # test set ground truth
    true_value = all_data['test']['target']
    # print(true_value.shape)


    train_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['train']['week']),
            torch.Tensor(all_data['train']['day']),
            torch.Tensor(all_data['train']['recent']),
            torch.Tensor(all_data['train']['target'])
        ),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )


    val_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['val']['week']),
            torch.Tensor(all_data['val']['day']),
            torch.Tensor(all_data['val']['recent']),
            torch.Tensor(all_data['val']['target'])
        ),
        batch_size=batch_size,
        shuffle=False
    )


    test_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['test']['week']),
            torch.Tensor(all_data['test']['day']),
            torch.Tensor(all_data['test']['recent']),
            torch.Tensor(all_data['test']['target'])
        ),
        batch_size=batch_size,
        shuffle=False
    )
    return true_value, train_loader, val_loader, test_loader
