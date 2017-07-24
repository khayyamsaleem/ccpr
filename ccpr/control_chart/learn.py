import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

"""Hyperparameters"""
num_filt_1 = 16 #Number of filters in first conv layer
num_filt_2 = 14
num_filt_3 = 8
num_fc_1 = 40 #number of neurons in fully connected layer
max_iterations = 20000
batch_size = 64
dropout = 1.0
learning_rate = 2e-5
input_norm = False

"""loading data"""

