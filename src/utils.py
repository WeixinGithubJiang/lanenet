import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import cv2

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
    for i, lane in enumerate(pts):
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


def line_accuracy(pred, gt, thresh):
    pred = np.array([p if p >= 0 else -100 for p in pred])
    gt = np.array([g if g >= 0 else -100 for g in gt])
    return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)


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

        # convert to probabiblity output to cal precision/recall
        # preds = F.sigmoid(preds)
        #val_score.update(preds.data.cpu().numpy(), val_data[1].numpy())

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


def average_precision(pred, label):
    """calculate average precision
    for each relevant label, average precision computes the proportion
    of relevant labels that are ranked before it, and finally averages
    over all relevant labels [1]

    References:
    ----------
    .. [1] Sorower, Mohammad S. "A literature survey on algorithms for
    multi-label learning." Oregon State University, Corvallis (2010).

    Notes:
    -----
    .. Check with the average_precision_score method in the sklearn.metrics package
    average_precision_score(pred, label, average='samples')

    """
    ap = 0
    # sort the prediction scores in the descending order
    sorted_pred_idx = np.argsort(pred)[::-1]
    ranks = np.empty(len(pred), dtype=int)
    ranks[sorted_pred_idx] = np.arange(len(pred)) + 1

    # only care of those ranks of relevant labels
    ranks = ranks[label > 0]

    for ii, rank in enumerate(sorted(ranks)):
        num_relevant_labels = ii + 1  # including the current relevant label
        ap = ap + float(num_relevant_labels) / rank

    return 0 if len(ranks) == 0 else ap / len(ranks)


class AverageScore(object):
    """Compute precision/recall/f-score and mAP"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.threshold_values = list(np.arange(0.1, 1, 0.1))
        self.num_correct = [0] * len(self.threshold_values)
        self.num_pred = [0] * len(self.threshold_values)
        self.num_gold = 0
        self.num_samples = 0
        self.sum_ap = 0

    def update(self, preds, labels):
        batch_size = preds.shape[0]

        self.num_samples += batch_size
        ap = 0
        for i in range(batch_size):
            pred = preds[i]
            label = labels[i]

            correct_pred = pred[label > 0]
            self.num_gold = self.num_gold + len(np.nonzero(label)[0])

            for j, t in enumerate(self.threshold_values):
                self.num_pred[j] = self.num_pred[j] + len(pred[pred > t])
                self.num_correct[j] = self.num_correct[
                    j] + len(correct_pred[correct_pred > t])

            ap += average_precision(pred, label)

        self.sum_ap += ap

    def map(self):
        return 0 if self.num_samples == 0 else self.sum_ap / self.num_samples

    def __str__(self):
        """String representation for logging
        """
        out = ''
        for i, t in enumerate(self.threshold_values):
            p = 0 if self.num_pred[i] == 0 else float(
                self.num_correct[i]) / self.num_pred[i]
            r = 0 if self.num_gold == 0 else float(
                self.num_correct[i]) / self.num_gold
            f = 0 if p + r == 0 else 2 * p * r / (p + r)
            out += '===> Precision = %.4f, Recall = %.4f, F-score = %.4f (@ threshold = %.1f)\n' % (
                p, r, f, t)
        out += '===> Mean AP = %.4f' % (self.sum_ap / self.num_samples)
        return out


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

