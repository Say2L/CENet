import torch

DEVICE = torch.device("cuda:0")

"""
# MOSEI SETTING
ACOUSTIC_DIM = 74
VISUAL_DIM = 35
TEXT_DIM = 768
"""

# MOSI SETTING
ACOUSTIC_DIM = 74
VISUAL_DIM = 27
TEXT_DIM = 768

ROBERTA_INJECTION_INDEX = 1

input_size = VISUAL_DIM # ACOUSTIC_DIM  VISUAL_DIM
hidden_size = 768
ffn_num_hiddens = 1024
max_sequence_len = 50
num_head = 8
label_size = 16

head_dropout = 0.1,
head_hidden_dim = 64 
