# IN_DIR=/nas/datashare/datasets/tusimple-benchmark
IN_DIR=/home/sang/datasets/tusimple-benchmark
OUT_DIR=output
META_DIR=$(OUT_DIR)/metadata
MODEL_DIR=$(OUT_DIR)/model
BIN_DIR=$(OUT_DIR)/bin_images # contains binary segmentation images
INS_DIR=$(OUT_DIR)/ins_images # contains instance segmentation images

# Variables
TEST_FILE=$(IN_DIR)/test_tasks_0627.json
GT_FILE=$(IN_DIR)/test_label.json
TEST_IMG_DIR?=/home/sang/datasets/20180914
MODEL_FILE?=$(MODEL_DIR)/lanenet.pth
PRED_FILE=$(MODEL_FILE:.pth=_predictions.json)

# Download data , tips: type `make -j3 download` to download three parts simultenously. 
download: $(IN_DIR)/train_set.zip $(IN_DIR)/test_baseline.json $(IN_DIR)/test_set.zip $(IN_DIR)/test_label.json
$(IN_DIR)/train_set.zip:
	wget https://tusimple-benchmark-evaluation.s3.amazonaws.com/datasets/1/train_set.zip -P $(IN_DIR) 
$(IN_DIR)/test_baseline.json:
	wget https://tusimple-benchmark-evaluation.s3.amazonaws.com/datasets/1/test_baseline.json -P $(IN_DIR) 
$(IN_DIR)/test_set.zip:
	wget https://tusimple-benchmark-evaluation.s3.amazonaws.com/datasets/1/test_set.zip -P $(IN_DIR)
$(IN_DIR)/test_label.json:
	wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/truth/1/test_label.json -P $(IN_DIR)

# extract the data 
# please go to the output directory and extract the images by yourself.
# for example, upzip train_set.zip 


# For TuSimple dataset,
# Merge all the annotation and create train/val splits

SPLITS=train val test

metadata: $(META_DIR)/tusimple.json
$(META_DIR)/tusimple.json:
	python src/metadata.py --input_dir $(IN_DIR) --output_file $@

# Generate binary segmentation image & instance segmentation images from
# the annotation data
generate_label_images:
	python src/gen_seg_images.py $(META_DIR)/tusimple.json $(IN_DIR) \
		--bin_dir $(BIN_DIR) \
		--ins_dir $(INS_DIR) \
		--splits train val \
		--thickness 5

train: $(MODEL_FILE)
$(MODEL_FILE): $(META_DIR)/tusimple.json 
	python src/train.py $^ $@ \
		--image_dir $(IN_DIR) \
		--batch_size 2 \
		--cnn_type unet 

test_tusimple: $(PRED_FILE)
$(PRED_FILE): $(MODEL_FILE) $(TEST_FILE) 
	python src/test.py $< \
		--output_file $@ \
		--meta_file $(word 2, $^) \
		--image_dir $(IN_DIR) \
		--loader_type tutest \
		--batch_size 3

# The provided evaluation script was written in Python 2, while this project use Python 3
# Solution is to use an Python 2 env for the evaluation
# Makefile uses /bin/sh as the default shell, which does not implement source
# change SHELL to /bin/bash to activate the Python 2 environment
SHELL=/bin/bash 
eval_tusimple: $(PRED_FILE) $(GT_FILE) 
	source activate py2 && \
		python tusimple-benchmark/evaluate/lane.py $^ && \
	source deactivate


# Show the results for each image on the test set (by turning on the show_demo switch)
demo_tusimple: $(MODEL_FILE) $(META_DIR)/tusimple.json
	python src/test.py $< \
		--meta_file $(word 2, $^) \
		--image_dir $(IN_DIR) \
		--save_dir $(OUT_DIR)/demo_tusimple \
		--loader_type meta \
		--batch_size 1 --show_demo


# Examples of make rules to test lane detection from an image directory
test_dir: $(MODEL_FILE) 
	python src/test.py $^ \
		--image_dir $(TEST_IMG_DIR)  \
		--save_dir $(OUT_DIR)/test_dir \
		--loader_type dir \
		--image_ext png \
		--batch_size 1 

test_0: $(MODEL_FILE) 
	python src/test.py $^ \
		--image_dir /users/yizhou/rosbags/top_down_view_university_back_0 \
		--save_dir $(OUT_DIR)/top_down_view_university_back_0 \
		--loader_type dir \
		--image_ext png \
		--batch_size 1 

test_2: $(MODEL_FILE) 
	python src/test.py $^ \
		--image_dir /users/yizhou/rosbags/top_down_view_university_back_2 \
		--save_dir $(OUT_DIR)/top_down_view_university_back_2 \
		--loader_type dir \
		--image_ext png \
		--batch_size 1 

test_4: $(MODEL_FILE) 
	python src/test.py $^ \
		--image_dir /users/yizhou/rosbags/top_down_view_university_back_4 \
		--save_dir $(OUT_DIR)/top_down_view_university_back_4 \
		--loader_type dir \
		--image_ext png \
		--batch_size 1 
