import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable

import logging
logger = logging.getLogger(__name__)

from unet import UNet

class LaneNet(nn.Module):

    def __init__(
            self,
            cnn_type='unet',
            pretrained=True):
        """Load a pretrained model and replace top fc layer."""
        super(LaneNet, self).__init__()

        self.core = self.get_cnn(cnn_type, pretrained)


    def get_cnn(self, cnn_type, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        logger.info("===> Loading pre-trained model '{}'".format(cnn_type))

        if cnn_type == 'unet':
            model = UNet()
        else:
            raise ValueError('cnn_type unknown: %s', cnn_type)

        return model


    def forward(self, images):
        """Extract image feature vectors."""

        out = self.core(images)

        return out
