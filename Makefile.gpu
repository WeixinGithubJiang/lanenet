
DATASET_DIR=/datashare/datasets_3rd_party
DATASETS=tusimple culane
DATASET=tusimple

TUSIMPLE_DATA_DIR=$(DATASET_DIR)/tusimple-benchmark
CULANE_DATA_DIR=$(DATASET_DIR)/CULane

ifeq ($(DATASET), tusimple)
	DATA_DIR=$(TUSIMPLE_DATA_DIR)
else ifeq ($(DATASET), culane)
	DATA_DIR=$(CULANE_DATA_DIR)
else
    @echo 'Unknown $(DATASET)!!!'
endif

OUT_DIR=/datashare/users/sang/works/lanenet/output
META_DIR=$(OUT_DIR)/metadata
MODEL_DIR=$(OUT_DIR)/model

# Variables
TEST_FILE=$(DATA_DIR)/test_tasks_0627.json
GT_FILE=$(DATA_DIR)/test_label.json
TEST_IMG_DIR?=/home/sang/datasets/20180914

MODEL_FILE=$(MODEL_DIR)/$(DATASET).pth
PRED_FILE=$(MODEL_FILE:.pth=_predictions.json)

BATCH_SIZE?=16

# Download data , tips: type `make -j3 download` to download three parts simultenously. 
download: $(DATA_DIR)/train_set.zip $(DATA_DIR)/test_baseline.json $(DATA_DIR)/test_set.zip $(DATA_DIR)/test_label.json
$(DATA_DIR)/train_set.zip:
	wget https://tusimple-benchmark-evaluation.s3.amazonaws.com/datasets/1/train_set.zip -P $(DATA_DIR) 
$(DATA_DIR)/test_baseline.json:
	wget https://tusimple-benchmark-evaluation.s3.amazonaws.com/datasets/1/test_baseline.json -P $(DATA_DIR) 
$(DATA_DIR)/test_set.zip:
	wget https://tusimple-benchmark-evaluation.s3.amazonaws.com/datasets/1/test_set.zip -P $(DATA_DIR)
$(DATA_DIR)/test_label.json:
	wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/truth/1/test_label.json -P $(DATA_DIR)

# For TuSimple dataset,
# Merge all the annotation and create train/val splits

SPLITS=train val test

metadata: $(patsubst %,$(META_DIR)/%.json,$(DATASETS))
$(META_DIR)/tusimple.json:
	python src/metadata.py --input_dir $(TUSIMPLE_DATA_DIR) \
		--dataset tusimple \
		--output_file $@
$(META_DIR)/culane.json:
	python src/metadata.py --input_dir $(CULANE_DATA_DIR) \
		--dataset culane \
		--output_file $@

# Generate binary segmentation image & instance segmentation images from
# the annotation data
BIN_DIR=$(OUT_DIR)/bin_images # contains binary segmentation images
INS_DIR=$(OUT_DIR)/ins_images # contains instance segmentation images
generate_label_images:
	python src/gen_seg_images.py $(META_DIR)/tusimple.json $(DATA_DIR) \
		--bin_dir $(BIN_DIR) \
		--ins_dir $(INS_DIR) \
		--splits train val \
		--thickness 5

train: $(MODEL_FILE)
$(MODEL_FILE): $(META_DIR)/$(DATASET).json 
	python src/train.py $^ $@ \
		--image_dir $(DATA_DIR) \
		--batch_size $(BATCH_SIZE) \
		--cnn_type unet \
		--dataset $(DATASET) \
		--thickness 10

test_tusimple: $(PRED_FILE)
$(PRED_FILE): $(MODEL_FILE) $(TEST_FILE) 
	python src/test.py $< \
		--output_file $@ \
		--meta_file $(word 2, $^) \
		--image_dir $(DATA_DIR) \
		--loader_type tutest \
		--batch_size $(BATCH_SIZE)

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
		--image_dir $(DATA_DIR) \
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
