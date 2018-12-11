from .tusimple import TuSimpleDataLoader
from .culane import CULaneDataLoader
from .dirloader import DirDataLoader

import torch.utils.data as data

datasets = {
	'tusimple': TuSimpleDataLoader,
	'culane': CULaneDataLoader,
	'dirloader': DirDataLoader,
}

def get_dataset(opt, **kwargs):
    loader_type = kwargs['loader_type'] if 'loader_type' in kwargs else None
    if loader_type == 'dirloader':
        key = 'dirloader'
    else:
        key = opt.dataset
    return datasets[key](opt, **kwargs)

def get_data_loader(opt, **kwargs):

    dataset = get_dataset(opt, **kwargs)

    shuffle = kwargs['split'] == 'train'
    data_loader = data.DataLoader(dataset,
                                  batch_size=opt.batch_size,
                                  num_workers=opt.num_workers,
                                  shuffle=shuffle,
                                  pin_memory=True)
    return data_loader
