#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

### YOUR CODE HERE for part 1d
class Highway(nn.Module):
  def __init__(self, word_emb_size):
    super(Highway, self).__init__()
    self.projection_layer = nn.Linear(word_emb_size, word_emb_size)
    self.gate_layer = nn.Linear(word_emb_size, word_emb_size)

  def forward(self, x_conv_out):
    # x_conv_out shape: (max_sent_len, batch_size, word_emb_size)
    x_proj = F.relu(self.projection_layer(x_conv_out)) # shape: (max_sent_len, batch_size, word_emb_size)
    x_gate = torch.sigmoid(self.gate_layer(x_conv_out)) # shape: (max_sent_len, batch_size, word_emb_size)
    x_highway = x_gate*x_proj + (1-x_gate)*x_conv_out # shape: (max_sent_len, batch_size, word_emb_size)
    return x_highway

### END YOUR CODE
