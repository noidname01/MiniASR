from glob import glob
from string import ascii_lowercase
from collections import defaultdict
import pickle
import numpy as np
from miniasr.prefix_beam_search import prefix_beam_search

class LanguageModel(object):
  """
  Loads a dictionary mapping between prefixes and probabilities.
  """

  def __init__(self):
    """
    Initializes the language model.

    Args:
      lm_file (str): Path to dictionary mapping between prefixes and lm probabilities. 
    """
    lm = pickle.load(open("./language_model.p", 'rb'))
    self._model = defaultdict(lambda: 1e-11, lm)

  def __call__(self, prefix):
    """
    Returns the probability of the last word conditioned on all previous ones.

    Args:
      prefix (str): The sentence prefix to be scored.
    """
    return self._model[prefix]
