import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
import math
import json

from datetime import datetime
from model import DepNet
from dataloader import get_data_loader
from utils import test

import logging
logger = logging.getLogger(__name__)


def main(opt):
    logger.info('Loading model: %s', opt.model_file)

    test_opt = {
        'label_file': opt.test_label,
        'imageinfo_file': opt.test_imageinfo,
        'image_dir': opt.test_image_dir,
        'batch_size': opt.batch_size,
        'num_workers': opt.num_workers,
        'train': False
    }

    checkpoint = torch.load(opt.model_file)

    test_loader = get_data_loader(test_opt)
    num_labels = test_loader.dataset.get_num_labels()

    logger.info('Building model...')
    checkpoint_opt = checkpoint['opt']
    model = DepNet(
        num_labels,
        finetune=checkpoint_opt.finetune,
        cnn_type=checkpoint_opt.cnn_type,
        pretrained=False)

    criterion = nn.MultiLabelSoftMarginLoss()
    model.load_state_dict(checkpoint['model'])

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    logger.info('Start testing...')
    test_loss, test_score = test(checkpoint_opt, model, criterion, test_loader)
    logger.info('Test loss: \n%s', test_loss)
    logger.info('Test score: \n%s', test_score)

    out = {'map': test_score.map()}
    logger.info('Writing output to %s', opt.output_file)
    with open(opt.output_file, 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'test_label',
        type=str,
        help='path to the h5 file containing the testing labels info')
    parser.add_argument(
        'test_imageinfo',
        type=str,
        help='imageinfo contains image path')
    parser.add_argument('model_file', type=str, help='path to the model file')
    parser.add_argument(
        'output_file',
        type=str,
        help='path to the output file')
    parser.add_argument(
        '--test_image_dir',
        type=str,
        help='path to the image dir')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='batch size')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='number of workers (each worker use a process to load a batch of data)')
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

    opt = parser.parse_args()

    logging.basicConfig(level=getattr(logging, opt.loglevel.upper()),
                        format='%(asctime)s:%(levelname)s: %(message)s')

    logger.info(
        'Input arguments: %s',
        json.dumps(
            vars(opt),
            sort_keys=True,
            indent=4))

    if not os.path.isfile(opt.model_file):
        logger.info('Model file does not exist: %s', opt.model_file)

    else:
        start = datetime.now()
        main(opt)
        logger.info('Time: %s', datetime.now() - start)
        