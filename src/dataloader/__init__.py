from .tusimple import TuSimpleDataLoader
from .culane import CULaneDataLoader

import torch.utils.data as data

datasets = {
	'tusimple': TuSimpleDataLoader,
	'culane': CULaneDataLoader,
}


def get_dataset(opt, **kwargs):
    return datasets[opt.dataset](opt, **kwargs)

def get_data_loader(opt, **kwargs):

    dataset = get_dataset(opt, **kwargs)

    shuffle = kwargs['split'] == 'train'
    data_loader = data.DataLoader(dataset,
                                  batch_size=opt.batch_size,
                                  num_workers=opt.num_workers,
                                  shuffle=shuffle,
                                  pin_memory=True)
    return data_loader
