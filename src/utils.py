import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import cv2
from sklearn.cluster import MeanShift

import logging
logger = logging.getLogger(__name__)


def get_binary_image(img, pts, thickness=5):
    """ Get the binary image

    Args:
        img: numpy array
        pts: set of lanes, each lane is a set of points

    Output:

    """
    #import pdb; pdb.set_trace()
    bin_img = np.zeros(shape=[img.shape[0], img.shape[1]], dtype=np.uint8)
    for i, lane in enumerate(pts):
        cv2.polylines(bin_img, np.int32([lane]), isClosed=False, color=255, thickness=thickness)

    return bin_img

def get_instance_image(img, pts, thickness=5):
    """  Get the instance segmentation images,
    where each lane is annotated using a polyline with a
    different color

    Args:
            image
            pts

    Output:
            instance segmentation image
    """
    ins_img = np.zeros(shape=[img.shape[0], img.shape[1]], dtype=np.uint8)
    nlanes = len(pts)
    color_codes = list(range(0, 255, 255//(nlanes + 1)))[1:]

    for i, lane in enumerate(pts):
        cv2.polylines(ins_img, np.int32([lane]), isClosed=False, color=color_codes[i], thickness=thickness)

    return ins_img

def get_binary_labels(height, width, pts, thickness=5):
    """ Get the binary labels. this function is similar to
    @get_binary_image, but it returns labels in 2 x H x W format
    this label will be used in the CrossEntropyLoss function.

    Args:
        img: numpy array
        pts: set of lanes, each lane is a set of points

    Output:

    """
    bin_img = np.zeros(shape=[height, width], dtype=np.uint8)
    for lane in pts:
        cv2.polylines(bin_img, np.int32([lane]), isClosed=False, color=255, thickness=thickness)

    bin_labels = np.zeros_like(bin_img, dtype=bool)
    bin_labels[bin_img != 0] = True
    bin_labels = np.stack([~bin_labels, bin_labels]).astype(np.uint8)
    return bin_labels

def get_instance_labels(height, width, pts, thickness=5, max_lanes=5):
    """  Get the instance segmentation labels.
    this function is similar to @get_instance_image,
    but it returns label in L x H x W format

    Args:
            image
            pts

    Output:
            max Lanes x H x W, number of actual lanes
    """
    if len(pts) > max_lanes:
        logger.warning('More than 5 lanes: %s', len(pts))
        pts = pts[:max_lanes]

    ins_labels = np.zeros(shape=[0, height, width], dtype=np.uint8)

    n_lanes = 0
    for lane in pts:
        ins_img = np.zeros(shape=[height, width], dtype=np.uint8)
        cv2.polylines(ins_img, np.int32([lane]), isClosed=False, color=1, thickness=thickness)

        # there are some cases where the line could not be draw, such as one
        # point, we need to remove these cases
        # also, if there is overlapping among lanes, only the latest lane is labeled
        if ins_img.sum() != 0:
            ins_labels[:, ins_img != 0] = 0
            ins_labels = np.concatenate([ins_labels, ins_img[np.newaxis]])
            n_lanes += 1

    if n_lanes < max_lanes:
        n_pad_lanes = max_lanes - n_lanes
        pad_labels = np.zeros(shape=[n_pad_lanes, height, width], dtype=np.uint8)
        ins_labels = np.concatenate([ins_labels, pad_labels])

    return ins_labels, n_lanes


def train(opt, model, criterion_disc, criterion_ce, optimizer, loader, epoch):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images, bin_labels, ins_labels, n_lanes = data

        images = Variable(images, volatile=False)
        bin_labels = Variable(bin_labels, volatile=False)
        ins_labels = Variable(ins_labels, volatile=False)

        if torch.cuda.is_available():
            images = images.cuda()
            bin_labels = bin_labels.cuda()
            ins_labels = ins_labels.cuda()

        bin_preds, ins_preds = model(images)

        _, bin_labels_ce = bin_labels.max(1)
        ce_loss = criterion_ce(bin_preds.permute(0,2,3,1).contiguous().view(-1,2),
                               bin_labels_ce.view(-1))

        disc_loss = criterion_disc(ins_preds, ins_labels, n_lanes)
        loss = ce_loss + disc_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)

        # Print log info
        if i % opt.log_step == 0:
            logger.info(
                'Epoch [{0}][{1}/{2}]\t'
                'Loss {3:0.7f}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(loader), loss.data[0],
                    batch_time=batch_time,
                    data_time=data_time))

        end = time.time()


def test(opt, model, criterion_disc, criterion_ce, loader):
    val_loss = AverageMeter()
    val_score = AverageScore()
    model.eval()

    for i, data in enumerate(loader):
        # Update the model
        images, bin_labels, ins_labels, n_lanes = data

        images = Variable(images, volatile=False)
        bin_labels = Variable(bin_labels, volatile=False)
        ins_labels = Variable(ins_labels, volatile=False)

        if torch.cuda.is_available():
            images = images.cuda()
            bin_labels = bin_labels.cuda()
            ins_labels = ins_labels.cuda()

        bin_preds, ins_preds = model(images)

        _, bin_labels_ce = bin_labels.max(1)
        ce_loss = criterion_ce(bin_preds.permute(0,2,3,1).contiguous().view(-1,2),
                               bin_labels_ce.view(-1))

        disc_loss = criterion_disc(ins_preds, ins_labels, n_lanes)
        loss = ce_loss + disc_loss

        val_loss.update(loss.data[0])

        if i % opt.log_step == 0:
            logger.info(
                'Epoch [{0}][{1}/{2}]\t'
                'Loss {3:0.7f}\t'.format(
                    0, i, len(loader), loss.data[0]))

    return val_loss #, val_score


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every [lr_update] epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


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
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(kernel_size, kernel_size))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        ccs = cv2.connectedComponentsWithStats(image, connectivity=8, ltype=cv2.CV_32S)
        labels = ccs[1]
        stats = ccs[2]

        for index, stat in enumerate(stats):
            if stat[4] <= minarea_threshold:
                idx = np.where(labels == index)
                image[idx] = 0

        return image


class Cluster(object):

    def __init__(self):
        pass

    def cluster(self, prediction, bandwidth=1.5):
        """
        :param prediction:
        :param bandwidth:
        :return:
        """
        ms = MeanShift(bandwidth, bin_seeding=True)
        tic = time.time()
        try:
            ms.fit(prediction)
        except ValueError as err:
            log.error(err)
            return 0, [], []
        # log.info('Mean Shift耗时: {:.5f}s'.format(time.time() - tic))
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        num_clusters = cluster_centers.shape[0]

        return num_clusters, labels, cluster_centers


def get_lane_area(binary_seg_ret, instance_seg_ret):
    """
    :param binary_seg_ret:
    :param instance_seg_ret:
    :return:
    """
    idx = np.where(binary_seg_ret == 1)

    lane_embedding_feats = []
    lane_coordinate = []
    for i in range(len(idx[0])):
        lane_embedding_feats.append(instance_seg_ret[:, idx[0][i], idx[1][i]])
        lane_coordinate.append([idx[0][i], idx[1][i]])

    return np.array(lane_embedding_feats, np.float32), np.array(lane_coordinate, np.int64)


def get_lane_mask(num_clusters, labels, binary_seg_ret, lane_coordinate):
    """
    :param binary_seg_ret:
    :param instance_seg_ret:
    :return:
    """

    color_map = [(255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (125, 125, 0),
                (0, 125, 125),
                (125, 0, 125),
                (50, 100, 50),
                (100, 50, 100)]

    ## continue working on this
    if num_clusters > 8:
        cluster_sample_nums = []
        for i in range(num_clusters):
            cluster_sample_nums.append(len(np.where(labels == i)[0]))
        sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))
        cluster_index = np.array(range(num_clusters))[sort_idx[0:8]]
    else:
        cluster_index = range(num_clusters)

    mask_image = np.zeros(shape=[binary_seg_ret.shape[0], binary_seg_ret.shape[1], 3], dtype=np.uint8)

    for index, i in enumerate(cluster_index):
        idx = np.where(labels == i)
        coord = lane_coordinate[idx]
        coord = np.flip(coord, axis=1)
        color = color_map[index]
        coord = np.array([coord])
        cv2.polylines(img=mask_image, pts=coord, isClosed=False, color=color, thickness=2)

    return mask_image

