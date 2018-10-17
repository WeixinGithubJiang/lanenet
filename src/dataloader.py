import os
import sys
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import json
import cv2
from utils import get_binary_labels, get_instance_labels
import matplotlib.pyplot as plt
import logging
from datetime import datetime
logger = logging.getLogger(__name__)


def get_image_transform(height=256, width=512):

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    t = [transforms.ToTensor(),
         normalizer]

    transform = transforms.Compose(t)
    return transform


class DataLoader(data.Dataset):
    """
    Load raw images and labels
    where labels are a binary map
    and an instance semegtation map
    """

    def __init__(self, opt, split='train'):

        self.image_dir = opt.image_dir
        self.thickness = opt.thickness
        self.height = opt.height
        self.width = opt.width
        self.max_lanes = opt.max_lanes
        self.split = split

        self.image_transform = get_image_transform(height=self.height, width=self.width)

        logger.info('Loading meta file: %s', opt.meta_file)

        data = json.load(open(opt.meta_file))
        self.info = data[split]

        self.image_ids = list(self.info.keys())

    def __getitem__(self, index):

        image_id = self.image_ids[index]
        file_name = self.info[image_id]['raw_file']
        file_path = os.path.join(self.image_dir, file_name)
        #image = Image.open(file_path).convert('RGB')
        image = cv2.imread(file_path) # in BGR order
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        width_org = image.shape[1]
        height_org = image.shape[0]

        image = cv2.resize(image, (self.width, self.height))

        x_lanes = self.info[image_id]['lanes']
        y_samples = self.info[image_id]['h_samples']
        pts = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in x_lanes]

        # remove empty lines (not sure the role of these lines, but it causes
        # bugs)
        pts = [l for l in pts if len(l) > 0]

        x_rate = 1.0*self.width/width_org
        y_rate = 1.0*self.height/height_org

        pts = [[(int(round(x*x_rate)), int(round(y*y_rate))) for (x, y) in lane] for lane in pts]

        # get the binary segmentation image and convert it into labels,
        # that has size 2 x Height x Weight
        bin_labels = get_binary_labels(self.height, self.width, pts,
                                       thickness=self.thickness)

        # get the instance segmentation image and convert it to labels
        # that has size Max_lanes x Height x Width
        ins_labels, n_lanes = get_instance_labels(self.height, self.width, pts,
                                         thickness=self.thickness,
                                         max_lanes=self.max_lanes)
        # transform the image, and convert to Tensor
        image_t = self.image_transform(image)
        bin_labels = torch.Tensor(bin_labels)
        ins_labels = torch.Tensor(ins_labels)

        if self.split == 'test':
            return image_t, image
        else:
            return image_t, bin_labels, ins_labels, n_lanes

    def __len__(self):
        return len(self.image_ids)

def collate_fn(data):
    """build a batch of data

    Note:
        probably the default collate_fn will do the same thing
    """
    images, bin_labels, ins_labels, n_lanes = zip(*data)
    images = torch.stack(images, 0)
    bin_labels = torch.stack(bin_labels, 0)
    ins_labels = torch.stack(ins_labels, 0)
    n_lanes = list(n_lanes)
    return images, bin_labels, ins_labels, n_lanes

def get_data_loader(opt, split='train'):
    """Returns torch.utils.data.DataLoader for custom dataset."""

    dataset = DataLoader(opt, split=split)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              pin_memory=False,
                                              num_workers=opt.num_workers,
                                              batch_size=opt.batch_size,
                                              #collate_fn=collate_fn,
                                              shuffle=split=='train')

    return data_loader
