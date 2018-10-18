

## path settings

# IN_DIR=/nas/datashare/datasets/tusimple-benchmark
IN_DIR=/home/sang/datasets/tusimple-benchmark
OUT_DIR=output
META_DIR=$(OUT_DIR)/metadata
MODEL_DIR=$(OUT_DIR)/model
BIN_DIR=$(OUT_DIR)/bin_images # contains binary segmentation images
INS_DIR=$(OUT_DIR)/ins_images # contains instance segmentation images
## variables


## rules

# download data , tips: type `make -j3 download` to download three parts simultenously. 

download: $(IN_DIR)/train_set.zip $(IN_DIR)/test_baseline.json $(IN_DIR)/test_set.zip
$(IN_DIR)/train_set.zip:
	wget https://tusimple-benchmark-evaluation.s3.amazonaws.com/datasets/1/train_set.zip -P $(IN_DIR) 
$(IN_DIR)/test_baseline.json:
	wget https://tusimple-benchmark-evaluation.s3.amazonaws.com/datasets/1/test_baseline.json -P $(IN_DIR) 
$(IN_DIR)/test_set.zip:
	wget https://tusimple-benchmark-evaluation.s3.amazonaws.com/datasets/1/test_set.zip -P $(IN_DIR)

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

train: $(MODEL_DIR)/lanenet.pth 
$(MODEL_DIR)/lanenet.pth: $(META_DIR)/tusimple.json 
	python src/train.py $^ $@ \
		--image_dir $(IN_DIR) \
		--batch_size 2 \
		--cnn_type unet 

#test: $(META_DIR)/tusimple.json $(MODEL_DIR)/lanenet_20181017.pth 
test: $(META_DIR)/tusimple.json $(MODEL_DIR)/lanenet.pth 
	python src/test.py $^ \
		--image_dir $(IN_DIR) \
		--batch_size 2 
