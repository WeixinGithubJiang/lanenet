import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import sys
import time
import math
import json
import logging

from datetime import datetime
from dataloader import get_data_loader
from model import LaneNet
logger = logging.getLogger(__name__)
from utils import AverageMeter, adjust_learning_rate
from loss import DiscriminativeLoss


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


def main(opt):

    # Set the random seed manually for reproducibility.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
    else:
        torch.manual_seed(opt.seed)

    train_loader = get_data_loader(opt, split='train')
    val_loader = get_data_loader(opt, split='val')

    logger.info('Building model...')

    model = LaneNet(cnn_type=opt.cnn_type)

    criterion_disc = DiscriminativeLoss(delta_var=0.5,
                                        delta_dist=1.5,
                                        norm=2,
                                        usegpu=True)

    criterion_ce = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

    if torch.cuda.is_available():
        model.cuda()
        criterion_disc.cuda()
        criterion_ce.cuda()

    logger.info("Start training...")
    best_loss = sys.maxsize
    best_epoch = 0

    for epoch in range(opt.num_epochs):
        learning_rate = adjust_learning_rate(opt, optimizer, epoch)
        logger.info('===> Learning rate: %f: ', learning_rate)

        # train for one epoch
        train(opt, model, criterion_disc, criterion_ce, optimizer, train_loader, epoch)

        # validate at every val_step epoch
        if epoch % opt.val_step == 0:
            logger.info("Start evaluating...")
            val_loss = test(opt, model, criterion_disc, criterion_ce, val_loader)
            logger.info('Val loss: \n%s', val_loss)
            #logger.info('Val score: \n%s', val_score)

            loss = val_loss.avg
            if loss < best_loss:
                logger.info(
                    'Found new best loss: %.7f, previous loss: %.7f',
                    loss,
                    best_loss)
                best_loss = loss
                best_epoch = epoch

                logger.info('Saving new checkpoint to: %s', opt.output_file)
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_loss': best_loss,
                    'best_epoch': best_epoch,
                    'opt': opt
                }, opt.output_file)

            else:
                logger.info(
                    'Current loss: %.7f, best loss is %.7f @ epoch %d',
                    loss,
                    best_loss,
                    best_epoch)

        if epoch - best_epoch > opt.max_patience:
            logger.info('Terminated by early stopping!')
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'meta_file',
        type=str,
        help='path to the metadata file containing train/val/test splits and image locations')

    parser.add_argument(
        'output_file',
        type=str,
        help='output model file (*.pth)')

    parser.add_argument(
        '--image_dir',
        type=str,
        help='path to image dir')

    # Model settings
    parser.add_argument(
        '--cnn_type',
        default='unet',
        choices=['unet'],
        help='The CNN used for image encoder (e.g. vgg19, resnet152)')

    # Optimization
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='batch size')

    parser.add_argument(
        '--width',
        type=int,
        default=512,
        help='image width to the network')

    parser.add_argument(
        '--height',
        type=int,
        default=256,
        help='image height to the network')

    parser.add_argument(
        '--thickness',
        type=int,
        default=5,
        help='thickness of the polylines')

    parser.add_argument(
        '--max_lanes',
        type=int,
        default=5,
        help='max number of lanes')

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='learning rate')

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=30,
        help='max number of epochs to run the training')
    parser.add_argument('--lr_update', default=10, type=int,
                        help='Number of epochs to update the learning rate.')

    parser.add_argument(
        '--max_patience',
        type=int,
        default=5,
        help='max number of epoch to run since the minima is detected -- early stopping')

    # other options
    parser.add_argument(
        '--val_step',
        type=int,
        default=1,
        help='how often do we check the model (in terms of epoch)')

    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='number of workers (each worker use a process to load a batch of data)')

    parser.add_argument(
        '--log_step',
        type=int,
        default=20,
        help='How often to print training info (loss, system/data time, etc)')

    parser.add_argument(
        '--loglevel',
        type=str,
        default='DEBUG',
        choices=[
            'DEBUG',
            'INFO',
            'WARNING',
            'ERROR',
            'CRITICAL'])

    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='random number generator seed to use')

    opt = parser.parse_args()

    logging.basicConfig(level=getattr(logging, opt.loglevel.upper()),
                        format='%(asctime)s:%(levelname)s: %(message)s')

    logger.info(
        'Input arguments: %s',
        json.dumps(
            vars(opt),
            sort_keys=True,
            indent=4))

    start = datetime.now()
    main(opt)
    logger.info('Time: %s', datetime.now() - start)
