import argparse
import os
import json
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime

import logging
logger = logging.getLogger(__name__)


def get_imageid(s):
    """Get image id from the raw_file path,
    e.g., clips/0313-1/8420/20.jpg'
    return: 0313-1-8420
    """
    p = s.split('/')
    img_id = p[1] + '-' + p[2]
    return img_id


def to_json(lines):
    """Convert list of json to json format
    """
    imgs = {}
    for l in lines:
        img_info = json.loads(l)
        img_id = get_imageid(img_info['raw_file'])
        if img_id in imgs:
            logger.error('Duplicated key %s', img_id)
        else:
            imgs[img_id] = img_info

    return imgs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_dir',
        type=str,
        help='Path to the dataset directory'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Path to the output file'
    )
    parser.add_argument(
        '--val_size',
        type=int,
        default=0.1,
        help='percentage of training image to be used as val data'
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s:%(levelname)s: %(message)s')

    logger.info(
        'Input arguments: %s',
        json.dumps(
            vars(args),
            sort_keys=True,
            indent=4))

    start = datetime.now()

    trainval_label_files = ['label_data_0313.json',
                            'label_data_0531.json',
                            'label_data_0601.json']

    test_label_file = 'test_tasks_0627.json'

    trainval_lines = []
    trainval_fileids = []
    n_train_images = 0
    for i, f in enumerate(trainval_label_files):
        label_file = os.path.join(args.input_dir, f)
        lines = [l for l in open(label_file, 'rb')]
        logger.info('Loaded %s images', len(lines))
        n_train_images += len(lines)
        trainval_lines.extend(lines)

        y = [i]*len(lines)
        trainval_fileids.extend(y)

    logger.info('Loaded %s training images', n_train_images)

    # this is to make sure val data is stratified over all annotation files
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=args.val_size,
        random_state=0)
    train_index, val_index = next(sss.split(trainval_lines, trainval_fileids))

    train_lines = [trainval_lines[i] for i in train_index]
    val_lines = [trainval_lines[i] for i in val_index]

    test_label_file = os.path.join(args.input_dir, test_label_file)
    test_lines = [l for l in open(test_label_file, 'rb')]
    logger.info('Loaded %s test images', len(test_lines))

    out = {}
    out['train'] = to_json(train_lines)
    out['val'] = to_json(val_lines)
    out['test'] = to_json(test_lines)

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json.dump(out, open(args.output_file, 'w'))
    logger.info('Saved output to %s', args.output_file)
    logger.info('Time: %s', datetime.now() - start)
