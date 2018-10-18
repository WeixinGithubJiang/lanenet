# LaneNet - Lane Detection Using Neural Networks

Pytorch implementation of lane detection networks. This is mainly based on the approach proposed in [Towards End-to-End Lane Detection: an Instance Segmentation Approach](https://arxiv.org/abs/1802.05591). This model simultaneously optimizes a binary semantic segmentation network using cross entropy loss, and a (lane) instance semantic segmentation using discriminative loss.

## Installation
This code has been tested on ubuntu 16.04(x64), python3.7, pytorch-0.4.1, cuda-9.0 with a GTX-1060 GPU. 
The Python environment can be imported from the `requirements.txt` file:
```
pip install -r requirements.txt
```

## Download data
- Edit the Makefile file and set the input directory $(IN_DIR) location. This is place where the dataset will be stored. If you already downloaded the data, then you can skip this step.
- Download [TuSimple dataset](https://github.com/TuSimple/tusimple-benchmark/wiki): `make download`.  Then extract the data.

## Generate train/val/test splits
- Run: `make matadata` to generate train/val/test split. Note that currently the test labels are not available, so we cannot do the quantitative evaluation yet. 

## Train model
- Run `make train`

```
usage: train.py [-h] [--image_dir IMAGE_DIR] [--cnn_type {unet}]
                [--batch_size BATCH_SIZE] [--width WIDTH] [--height HEIGHT]
                [--thickness THICKNESS] [--max_lanes MAX_LANES]
                [--learning_rate LEARNING_RATE] [--num_epochs NUM_EPOCHS]
                [--lr_update LR_UPDATE] [--max_patience MAX_PATIENCE]
                [--val_step VAL_STEP] [--num_workers NUM_WORKERS]
                [--log_step LOG_STEP]
                [--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--seed SEED]
                meta_file output_file
```

## Test model
- Run `make test`

```
usage: test.py [-h] [--image_dir IMAGE_DIR] [--batch_size BATCH_SIZE]
               [--num_workers NUM_WORKERS]
               [--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
               meta_file model_file
```

## Demo
   Check out the notebook to view groundtruth data [here](notebooks/view_groundtruth.ipynb), and to view examples of prediction results [here](notebooks/view_prediction.ipynb).

## Acknowledgements
- [Implemention of lanenet model for real time lane detection using deep neural network model](https://github.com/MaybeShewill-CV/lanenet-lane-detection)
- [Semantic Instance Segmentation with a Discriminative Loss Function in PyTorch](https://github.com/Wizaron/instance-segmentation-pytorch)
- [Implementation of discriminative loss for instance segmentation by pytorch](https://github.com/nyoki-mtl/pytorch-discriminative-loss)
