#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

from highway import Highway
from vocab import Vocab, VocabEntry

def q_1c(sents):
  v = Vocab.load("vocab_tiny_q2.json")
  tnsr = v.src.to_input_tensor_char(sents, "cpu")
  print (tnsr.shape)

def highway():
  torch.manual_seed(0)
  hwy = Highway(3)
  y = torch.tensor([[1,1,1], [0,1,0], [1,1,0], [2,3,4]], dtype=torch.float32)
  out = hwy(y)
  exp_out = torch.tensor([[0.6665, 0.3829, 0.6410],
        [0.2425, 0.4727, 0.1876],
        [0.7896, 0.4673, 0.1489],
        [1.8767, 0.3594, 2.7124]])

  print(out.data)
  print(torch.all(exp_out.eq(out.data)))
  print(exp_out.data)
  #assert(torch.all(exp_out.eq(out.data)))
  print("highway: Done")


def main():
    """ Main func.
    """
    sents = ["Hello how do you do good sir?".split(), "Couldn't be better Mr. McGiggles".split(), "I ought to say I am living the dream!".split()]
    q_1c(sents)
    highway()

if __name__ == '__main__':
    main()
