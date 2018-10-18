import numpy as np
import cv2
from sklearn.cluster import MeanShift
import torch.nn as nn
import logging

from unet import UNet

logger = logging.getLogger(__name__)


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


class PostProcessor(object):

    def __init__(self):
        pass

    def process(self, image, kernel_size=5, minarea_threshold=200):
        """

        :param image:
        :param kernel_size
        :param minarea_threshold
        :return:
        """
        if image.dtype is not np.uint8:
            image = np.array(image, np.uint8)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # fill the pixel gap using Closing operator (dilation followed by
        # erosion)
        kernel = cv2.getStructuringElement(
            shape=cv2.MORPH_RECT, ksize=(
                kernel_size, kernel_size))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        ccs = cv2.connectedComponentsWithStats(
            image, connectivity=8, ltype=cv2.CV_32S)
        labels = ccs[1]
        stats = ccs[2]

        for index, stat in enumerate(stats):
            if stat[4] <= minarea_threshold:
                idx = np.where(labels == index)
                image[idx] = 0

        return image


class LaneClustering(object):

    def __init__(self):
        pass

    def cluster(self, prediction, bandwidth=1.5):
        """
        :param prediction:
        :param bandwidth:
        :return:
        """
        ms = MeanShift(bandwidth, bin_seeding=True)
        try:
            ms.fit(prediction)
        except ValueError as err:
            logger.error(err)
            return 0, [], []

        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        num_clusters = cluster_centers.shape[0]

        return num_clusters, labels, cluster_centers
