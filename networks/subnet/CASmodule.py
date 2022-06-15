import torch
import torch.nn as nn
import torch.nn.functional as F
from config.global_configs import *
import math

class CAS(nn.Module):
    def __init__(self, beta_shift_a=0.5, beta_shift_v=0.5, dropout_prob=0.2):
        super(CAS, self).__init__()
        self.visual_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.acoustic_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.hv = SelfAttention(TEXT_DIM)
        self.ha = SelfAttention(TEXT_DIM)
        self.cat_connect = nn.Linear(2 * TEXT_DIM, TEXT_DIM)
        

    def forward(self, text_embedding, visual=None, acoustic=None, visual_ids=None, acoustic_ids=None):
        visual_ = self.visual_embedding(visual_ids)
        acoustic_ = self.acoustic_embedding(acoustic_ids)
        visual_ = self.hv(text_embedding, visual_)
        acoustic_ = self.ha(text_embedding, acoustic_) 
        visual_acoustic = torch.cat((visual_, acoustic_), dim=-1)
        shift = self.cat_connect(visual_acoustic)
        embedding_shift = shift + text_embedding
    
        return embedding_shift
        
class Attention(nn.Module):
    def __init__(self, text_dim):
        super(Attention, self).__init__()
        self.text_dim = text_dim
        self.dim = text_dim 
        self.Wq = nn.Linear(text_dim, text_dim)
        self.Wk = nn.Linear(self.dim, text_dim)
        self.Wv = nn.Linear(self.dim, text_dim)

    def forward(self, text_embedding, embedding):
        Q = self.Wq(text_embedding)
        K = self.Wk(embedding)
        V = self.Wv(embedding)
        tmp = torch.matmul(Q, K.transpose(-1, -2) * math.sqrt(self.text_dim))[0]
        weight_matrix = F.softmax(torch.matmul(Q, K.transpose(-1, -2) * math.sqrt(self.text_dim)), dim=-1)

        return torch.matmul(weight_matrix, V)


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, head_num=1):
        super(SelfAttention, self).__init__()
        self.head_num = head_num
        self.s_d = hidden_size // self.head_num
        self.all_head_size = self.head_num * self.s_d
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        x = x.view(x.size(0), x.size(1), self.head_num, -1)
        return x.permute(0, 2, 1, 3)

    def forward(self, text_embedding, embedding):
        Q = self.Wq(text_embedding)
        K = self.Wk(embedding)
        V = self.Wv(embedding)
        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)
        weight_score = torch.matmul(Q, K.transpose(-1, -2))
        weight_prob = nn.Softmax(dim=-1)(weight_score * 8)

        context_layer = torch.matmul(weight_prob, V)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
