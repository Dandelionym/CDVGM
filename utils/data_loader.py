import torch
from torch.utils.data import DataLoader, TensorDataset
from model.params import batch_size


def dataLoaders(all_data_):
	train_loader = DataLoader(
		TensorDataset(
			torch.Tensor(all_data_['train']['week']),
			torch.Tensor(all_data_['train']['day']),
			torch.Tensor(all_data_['train']['recent']),
			torch.Tensor(all_data_['train']['target'])
		),
		batch_size=batch_size,
		shuffle=False,
		pin_memory=True,
	)
	
	# validation set data loader
	val_loader = DataLoader(
		TensorDataset(
			torch.Tensor(all_data_['val']['week']),
			torch.Tensor(all_data_['val']['day']),
			torch.Tensor(all_data_['val']['recent']),
			torch.Tensor(all_data_['val']['target'])
		),
		batch_size=batch_size,
		shuffle=False,
		pin_memory=True,
	)
	
	# testing set data loader
	test_loader = DataLoader(
		TensorDataset(
			torch.Tensor(all_data_['test']['week']),
			torch.Tensor(all_data_['test']['day']),
			torch.Tensor(all_data_['test']['recent']),
			torch.Tensor(all_data_['test']['target'])
		),
		batch_size=batch_size,
		shuffle=False,
		pin_memory=True,
	)
	
	print("Train Loader - ")
	
	return train_loader, val_loader, test_loader
