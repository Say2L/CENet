# CENet
> Pytorch implementation for codes in "Cross-modal Enhancement Network for Multimodal Sentiment Analysis (TMM 2022)"(https://ieeexplore.ieee.org/document/9797846)
# Prepare
## Dataset
Download the MOSI pkl file (https://drive.google.com/drive/folders/1_u1Vt0_4g0RLoQbdslBwAdMslEdW1avI?usp=sharing). Put it under the ./dataset directory.

## Pre-trained language model
Download the SentiLARE language model files (https://drive.google.com/drive/folders/1u1YxwPGMcNDOmdnelPwBKYaBk-FY4jp1), and then put them into the ./pretrained-model/sentilare_model directory.

# Run
'''
python train.py
'''

Note: the scale of MOSI dataset is small, so the training process is not stable. To get results close to those in CENet paper, you can set the seed in args to 6758.
