

## path settings

IN_DIR=/nas/datashare/datasets/tusimple-benchmark
OUT_DIR=output

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



