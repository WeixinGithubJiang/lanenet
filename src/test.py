import argparse
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import logging

from model import LaneNet, PostProcessor, LaneClustering
from dataloader import get_data_loader
from utils import get_lane_area, get_lane_mask

logger = logging.getLogger(__name__)


def test(model, loader, postprocessor, clustering):
    model.eval()

    for data in loader:
        # Update the model
        images, _, _, _, org_images = data

        images = Variable(images, volatile=False)

        if torch.cuda.is_available():
            images = images.cuda()

        bin_preds, ins_preds = model(images)

        # convert to probabiblity output
        bin_preds = F.softmax(bin_preds, dim=1)
        # take the index of the max along the dim=1 dimension
        bin_preds = bin_preds.max(1)[1]

        bs = images.shape[0]
        for i in range(bs):
            bin_img = bin_preds[i].data.cpu().numpy()
            ins_img = ins_preds[i].data.cpu().numpy()

            bin_img = postprocessor.process(bin_img)

            lane_embedding_feats, lane_coordinate = get_lane_area(
                bin_img, ins_img)

            num_clusters, labels, cluster_centers = clustering.cluster(
                lane_embedding_feats, bandwidth=1.5)

            mask_img = get_lane_mask(num_clusters, labels, bin_img,
                                     lane_coordinate)

            plt.ion()
            plt.figure('mask_image')
            mask_img = mask_img[:, :, (2, 1, 0)]
            plt.imshow(mask_img)
            plt.figure('src_image')
            src_img = org_images[i]
            overlay_img = cv2.addWeighted(src_img, 1.0, mask_img, 1.0, 0)

            plt.imshow(overlay_img)
            plt.pause(3.0)
            plt.show()


def main(opt):
    logger.info('Loading model: %s', opt.model_file)

    checkpoint = torch.load(opt.model_file)

    checkpoint_opt = checkpoint['opt']
    model = LaneNet(cnn_type=checkpoint_opt.cnn_type)
    test_loader = get_data_loader(
        checkpoint_opt,
        split='test',
        return_raw_image=True)

    logger.info('Building model...')
    model.load_state_dict(checkpoint['model'])

    if torch.cuda.is_available():
        model.cuda()

    postprocessor = PostProcessor()
    clustering = LaneClustering()

    logger.info('Start testing...')
    test(model, test_loader, postprocessor, clustering)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'meta_file',
        type=str,
        help='path to the metadata file containing the testing labels info')
    parser.add_argument(
        'model_file',
        type=str,
        help='path to the model file')
    parser.add_argument(
        '--image_dir',
        type=str,
        help='path to the image dir')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='batch size')
    parser.add_argument(
        '--num_workers', type=int, default=0,
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
