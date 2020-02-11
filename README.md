# Capstone-TF2 Guide

## Setup Packages
 1. Install basic packages
> conda env create -f environment.yml
conda activate capstone-tf2

2. Install jupyter

> conda install -c anaconda jupyter

3. Install spyder
> conda install -c anaconda spyder=3.3.6

4. Install yaml

> pip install pyyaml==5.2

5. Install imageio

> conda install -c conda-forge imageio


## Files

1. Copy "ms1m_bin_in.tfrecord" to "data" folder
- Training dataset in tensorflow format 
2. Copy following folder files to "data\faces_test" folder
> agedb_align_112
cfp_align_112
lfw_align_112
- Testing dataset in aligned 112x112 image format
3. Edit "arc_res50_bin.yaml" file at "config" folder
- batch_size
-- If OOM (out of memory) error, decrease the batch size to 64, 32 or 16
-- Higher batch size reduces the training time
- save_steps
-- Control the checkpoints to be saved
-- Increase checkpoints will reduces the saved checkpoints files, but may lost the results
-- Only latest checkpoints or best checkpoints will be used for verification 

## Training and Verification

1. Training
- Open "train_arc_res50_bin.py" in Spyder
- Press "F5" to start training process
- Note:
-- When begin training process again, press "Ctrl + . " or select "Restart Kernel" to restart python kernel, then press "F5" to start new training process
2. Verification
- Copy best saved checkpoints files to "models" folder, and change their name to "arc_res50.ckpt.xxxx"
- Open "test_arc_res50_bin.py" in Spyder
- Press "F5" to start training process
- Take same note as training process
