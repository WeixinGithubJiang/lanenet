import argparse
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import logging
from tqdm import tqdm
import warnings

from torch.nn.parallel.scatter_gather import gather
from model import LaneNet, PostProcessor, LaneClustering
from dataloader import get_data_loader
from utils.utils import AverageMeter, get_lane_area, get_lane_mask, output_lanes
from utils.parallel import DataParallelModel

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def test(model, loader, postprocessor, clustering,
         show_demo=False, save_dir=None):
    """Test a model on image and display detected lanes

    Args:
        model (LaneNet): a LaneNet model
        loader (Dataloader) : data loader on test images
        postprocessor (PostProcessor): post processing, like filling empty gaps
            between nearby pixels using a closing operator
        clustering (LaneClustering): cluster lane embeddings to assign pixel
            to lane instance

    Returns:
        None

    """
    model.eval()

    run_time = AverageMeter()
    end = time.time()
    pbar = tqdm(loader)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for data in pbar:
            # Update the model
            images, org_images, image_ids = data

            images = Variable(images)

            if torch.cuda.is_available():
                images = images.cuda()

            bin_preds, ins_preds = gather(model(images), 0, dim=0)

            # convert to probabiblity output
            bin_preds = F.softmax(bin_preds, dim=1)
            # take the index of the max along the dim=1 dimension
            bin_preds = bin_preds.max(1)[1]

            bs = images.shape[0]
            for i in range(bs):
                bin_img = bin_preds[i].data.cpu().numpy()
                ins_img = ins_preds[i].data.cpu().numpy()
                image_id = image_ids[i]

                bin_img = postprocessor.process(bin_img)

                lane_embedding_feats, lane_coordinate = get_lane_area(
                    bin_img, ins_img)
                if lane_embedding_feats.size > 0:
                    num_clusters, labels, cluster_centers = clustering.cluster(
                        lane_embedding_feats, bandwidth=1.5)

                    mask_img = get_lane_mask(num_clusters, labels, bin_img,
                                            lane_coordinate)

                    mask_img = mask_img[:, :, (2, 1, 0)]
                    src_img = org_images[i].data.cpu().numpy()
                    overlay_img = cv2.addWeighted(src_img, 1.0, mask_img, 1.0, 0)
                else:
                    overlay_img = org_images[i].data.cpu().numpy()

                if show_demo:
                    plt.ion()
                    plt.figure('result')
                    plt.imshow(overlay_img)
                    plt.show()
                    plt.pause(0.01)

                if save_dir:
                    image_path = os.path.join(save_dir, image_id + '.' + opt.image_ext)
                    cv2.imwrite(image_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

            run_time.update(time.time() - end)
            end = time.time()
            fps = bs/run_time.avg
            pbar.set_description('Average run time: {fps:.3f} fps'.format(fps=fps))


def tusimpletest(model, loader, postprocessor, clustering):
    """Test a model on image and display detected lanes

    Args:
        model (LaneNet): a LaneNet model
        loader (Dataloader) : data loader on test images
        postprocessor (PostProcessor): post processing, like filling empty gaps
            between nearby pixels using a closing operator
        clustering (LaneClustering): cluster lane embeddings to assign pixel
            to lane instance

    Returns:
        None

    """
    model.eval()

    run_time = AverageMeter()
    pbar = tqdm(loader)

    x_lanes = []
    times = []
    with torch.no_grad():
        for data in pbar:
            # Update the model
            images, y_samples, widths, heights = data
            widths = widths.numpy()
            heights = heights.numpy()

            images = Variable(images)

            if torch.cuda.is_available():
                images = images.cuda()

            bin_preds, ins_preds = model(images)

            # convert to probabiblity output
            bin_preds = F.softmax(bin_preds, dim=1)
            # take the index of the max along the dim=1 dimension
            bin_preds = bin_preds.max(1)[1]

            bs, height, width = images.shape[0], images.shape[2], images.shape[3]

            for i in range(bs):
                end = time.time()
                bin_img = bin_preds[i].data.cpu().numpy()
                ins_img = ins_preds[i].data.cpu().numpy()

                bin_img = postprocessor.process(bin_img)

                lane_embedding_feats, lane_coordinate = get_lane_area(
                    bin_img, ins_img)

                num_clusters, labels, cluster_centers = clustering.cluster(
                        lane_embedding_feats, bandwidth=1.5)

                y_rate = 1.0*height/heights[i]
                x_rate = 1.0*width/widths[i]
                y_scaled = [y * y_rate for y in y_samples[i]]
                x_scaled = output_lanes(num_clusters, labels, bin_img, lane_coordinate, y_scaled)

                # project into original image size
                x_lanes_ = [[-2 if (x < 0 or x >= width) else int(round(x/x_rate)) for x in x_lane] for x_lane in x_scaled]
                x_lanes.append(x_lanes_)

                elapsed_time = time.time() - end

                # time should be reported in miliseconds here
                # if it is > 1 second, it will be evaluated at 0 score
                times.append(int(elapsed_time))
                run_time.update(elapsed_time)
            fps = 1.0/run_time.avg
            pbar.set_description('Average run time: {fps:.3f} fps'.format(fps=fps))

    return x_lanes, times

def output_tuprediction(test_file, x_lanes, times, output_file):
    test_lines = [l for l in open(opt.meta_file, 'rb')]
    logger.info('Loaded %s test images', len(test_lines))

    assert(len(test_lines) == len(x_lanes))
    info = []
    for i, l in enumerate(test_lines):
        img_info = json.loads(l)
        img_info['lanes'] = x_lanes[i]
        img_info['run_time'] = times[i]
        info.append(img_info)

    with open(output_file, 'w') as of:
        for img_info in info:
            json.dump(img_info, of)
            of.write('\n')

    logger.info('Wrote to %s', output_file)

def main(opt):
    logger.info('Loading model: %s', opt.model_file)

    checkpoint = torch.load(opt.model_file)

    checkpoint_opt = checkpoint['opt']

    # Load model location
    model = LaneNet(cnn_type=checkpoint_opt.cnn_type)
    model = DataParallelModel(model)

    # Update/Overwrite some test options like batch size, location to metadata
    # file
    vars(checkpoint_opt).update(vars(opt))

    test_loader = get_data_loader(
        checkpoint_opt,
        split='test',
        return_org_image=True)

    logger.info('Building model...')
    model.load_state_dict(checkpoint['model'])

    if torch.cuda.is_available():
        model = model.cuda()

    postprocessor = PostProcessor()
    clustering = LaneClustering()

    logger.info('Start testing...')

    if opt.loader_type == 'tusimpletest':
        x_lanes, times = tusimpletest(
            model,
            test_loader,
            postprocessor,
            clustering)
        output_tuprediction(opt.meta_file, x_lanes, times, opt.output_file)
    else:
        test(
            model,
            test_loader,
            postprocessor,
            clustering,
            show_demo=opt.show_demo,
            save_dir=opt.save_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_file',
        type=str,
        help='path to the model file')
    parser.add_argument(
        '--meta_file',
        type=str,
        help='path to the metadata file containing the testing labels info')
    parser.add_argument(
        '--output_file',
        type=str,
        help='path to the output file containing prediction info')
    parser.add_argument(
        '--image_dir',
        type=str,
        help='path to the image dir')
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='path to the save dir')
    parser.add_argument(
        '--image_ext',
        type=str,
        default='png',
        help='image extension, used to glob images based on its extension')
    parser.add_argument(
        '--loader_type',
        type=str,
        choices=['dataset', 'dirloader', 'tusimpletest'],
        default='dataset',
        help='data loader type, dir: from a directory; meta: from a metadata file')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='batch size')
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='number of workers (each worker use a process to load a batch of data)')
    parser.add_argument(
        '--show_demo', default=False, action='store_true',
        help='whether to show output image or not. If not, the running time \
        (fps) will be measured.')
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
