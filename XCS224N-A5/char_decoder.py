#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()

        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        pad_token_idx = target_vocab.char2id['<pad>']
        vocab_size = len(target_vocab.char2id)
        print("......vocab size ", vocab_size)
        print("......char embedding size ", char_embedding_size)

        self.char_output_projection = nn.Linear(hidden_size, vocab_size)
        self.decoderCharEmb = nn.Embedding(vocab_size, char_embedding_size, padding_idx=pad_token_idx)
        self.target_vocab = target_vocab
        self.loss = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum')

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        # Fetch the char embedding for the input
        input_embedding = self.decoderCharEmb(input) # shape: (length, batch, char_embed_size)
        output, dec_hidden = self.charDecoder(input_embedding, dec_hidden)
        scores = self.char_output_projection(output)
        return scores, dec_hidden
        ### END YOUR CODE


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        # train_forward(...) is a separate function because then we can run character level decoder for every word while
        # training. However this is only ever run for <unk> token during beam_search (aka inference time).
        #print ("dec_hidden\n", dec_hidden)
        #print ("char seq\n", char_sequence)
        # Get tokens from x_1 to x_n as described in hanout. In otw chop off the <END> token.
        char_sequence_1_n = char_sequence[0:-1, :] # shape : (length-1, batch)
        scores, char_dec_hidden = self.forward(char_sequence_1_n, dec_hidden)
        #print("scores ", scores.shape)
        #print("char_seq ", char_sequence[1:, :].shape)
        #print("char_seq_or ", char_sequence.shape)
        scores_flatten = scores.contiguous().view(-1,*scores.shape[2:]) # shape : (length*batch, vocab_size)
        # Get tokens from x_2 to x_n+1 to set as the target. (based on pdf description)
        char_sequence_2_n_plus_1 = char_sequence[1:, :] # shape : (length-1, batch)
        # Flatten the target character sequence to 1D with target character ids (length-1) * batch
        char_sequence_2_n_plus_1_flatten = \
            char_sequence_2_n_plus_1.contiguous().view(-1,*char_sequence_2_n_plus_1.shape[2:])
        cross_entropy_loss = self.loss(scores_flatten, char_sequence_2_n_plus_1_flatten)
        # print("CEL ", cross_entropy_loss)
        return cross_entropy_loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_size = initialStates[0].shape[1]
        hidden_size = initialStates[0].shape[2]
        #print ("batch size: ", batch_size, " hidden_size ", hidden_size)
        dec_hidden = initialStates
        output_word = [""] * batch_size
        current_char_id = torch.ones((1, batch_size), device=device, dtype=torch.long) \
                       * self.target_vocab.start_of_word # shape: (1, batch_size)
        for i in range(max_length):
            current_char_emb = self.decoderCharEmb(current_char_id)  # shape: (1, batch_size, char_emb_size)
            #print("current_char_emb_s ", current_char_emb.shape)
            outputs, dec_hidden = self.charDecoder(current_char_emb, dec_hidden)
            # dec_hidden - tuple of two tensors of shape: (1, batch, hidden_size)
            #print("outputs_s ", outputs.shape)

            scores = self.char_output_projection(outputs) # shape: (1, batch, self.vocab_size)
            p_t = nn.functional.softmax(scores, dim=-1) # shape: (1, batch, self.vocab_size)
            #print("p_t_s ", p_t.shape)

            current_char_id = torch.argmax(p_t, dim=-1) # shape: (1, batch)
            #print("current_char_s ", current_char_id.shape)
            #print("curr char ", current_char_id)
            output_word = [w + self.target_vocab.id2char[c.item()]
                           for c, w in zip(current_char_id.view(-1), output_word)]
            #output_word = [w+self.target_vocab.id2char[c.item()]
            #             if c != self.target_vocab.end_of_word else w
            #             for c, w in zip(current_char_id.view(-1 ), output_word)]
        #print("output_word ", output_word)
        # Truncate the words based on end of word token.
        output_word = [word.split(sep="}", maxsplit=1)[0] for word in output_word]
        #print("output_word ", output_word)

        #print("output_word_s ", len(output_word))
        return output_word
        ### END YOUR CODE

