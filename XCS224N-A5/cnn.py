#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

### YOUR CODE HERE for part 1d
class CNN(nn.Module):
  def __init__(self, char_emb_size, word_emb_size):
    super(CNN, self).__init__()
    self.conv_layer = nn.Conv1d(char_emb_size, word_emb_size, kernel_size=5)

  def forward(self, x_emb):
    # x_emb shape: (max_sent_len, batch_size, max_word_len, char_emb_size)
    # Pytorch does convolutions in the last dimension, therefore transpose the last 2 dimensions
    #
    max_sent_len = x_emb.shape[0]
    batch_size = x_emb.shape[1]
    #print("xemb_s ", x_emb.shape)
    x_reshaped = torch.transpose(x_emb, 2, 3) # shape: (max_sent_len, batch_size, char_emb_size, max_word_len)
    #print("x_reshaped_s ", x_reshaped.shape)
    # Flatten the batch_size and max_sentence_length into one dim to apply the  1D convolution.
    x_reshaped_flatten = torch.reshape(x_reshaped, (max_sent_len * batch_size, x_reshaped.shape[2], x_reshaped.shape[3]))
    #print("x_reshaped_flatten_s ", x_reshaped_flatten.shape)
    x_conv = F.relu(self.conv_layer(x_reshaped_flatten)) # shape: (max_sent_len * batch_size, word_emb_size, some_num_of_windows)
    #print("x_conv_s ", x_conv.shape)

    # MaxPool over all the windows in the word (last dim) to obtain a word_emb sized vector for each word.
    x_conv, _ = torch.max(x_conv, 2) # shape: (max_sent_len * batch_size, word_emb_size)
    #print("x_conv_s ", x_conv.shape)

    x_conv_out = torch.reshape(x_conv, (max_sent_len, batch_size, x_conv.shape[1]))
    #print("x_conv_out_s ", x_conv_out.shape)

    return x_conv_out

### END YOUR CODE
