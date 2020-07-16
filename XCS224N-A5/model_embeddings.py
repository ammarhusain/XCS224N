#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        char_embed_size = 50
        word_embed_size = embed_size
        self.embed_size = embed_size
        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), char_embed_size, padding_idx=pad_token_idx)
        self.conv_step = CNN(char_embed_size, word_embed_size)
        self.highway_net = Highway(word_embed_size)
        self.dropout = nn.Dropout(p=0.3)
        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        output = self.embeddings(input_tensor) # shape: (max_sent_len, batch_size, max_word_len, char_emb_len)
        # Run this through a 1d CNN
        x_conv_out = self.conv_step(output) # shape: (max_sent_len, batch_size, word_emb_size)
        x_highway_out = self.highway_net(x_conv_out) # shape: (max_sent_len, batch_size, word_emb_size)
        x_word = self.dropout(x_highway_out)
        return x_word
        ### END YOUR CODE
