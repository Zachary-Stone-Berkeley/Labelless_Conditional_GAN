import numpy as np
import sklearn
import tensorflow as tf
import h5py
from sklearn.preprocessing import scale

class DataSet(object):

  def __init__(self, 
               list_of_data, # a numpy array of targets
               augmentor=None):
    
    self.list_of_data = list_of_data
    self.epochs_completed = 0
    self.index_in_epoch = 0
    self.num_examples = list_of_data[0].shape[0]
    self.augmentor = augmentor
    self.num_data = len(self.list_of_data)
    
    for i in range(len(list_of_data)):
      assert list_of_data[i].shape[0] == self.num_examples, (
      "shape of data %s: %s, shape of data 0: %s" % (i, list_of_data[i].shape, self.num_examples))
  
  def next_batch(self, 
                 batch_size, 
                 shuffle=True,
                 use_aug=True):
    # the first observation of the new minibatch
    start = self.index_in_epoch
    # shuffle if this is the first minibatch
    if self.epochs_completed == 0 and start == 0 and shuffle:
      perm = np.arange(self.num_examples)
      np.random.shuffle(perm)
      for i in range(self.num_data):
        self.list_of_data[i] = self.list_of_data[i][perm]
    # if we have completed this epoch...
    if start + batch_size > self.num_examples:
      self.epochs_completed += 1
      # get the remaining observations
      rest_num_examples = self.num_examples - start
      rest_part = [self.list_of_data[i][start:self.num_examples] for i in range(self.num_data)]
      # shuffle the observations
      if shuffle:
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        for i in range(self.num_data):
          self.list_of_data[i] = self.list_of_data[i][perm]
      # get new observations to complete minibatch
      start = 0      
      self.index_in_epoch = batch_size - rest_num_examples
      end = self.index_in_epoch
      new_part = [self.list_of_data[i][start:end] for i in range(self.num_data)]      
      return [np.concatenate((rest_part[i], new_part[i]), axis=0) for i in range(self.num_data)]

    # if we haven't completed this epoch...
    else:
      self.index_in_epoch += batch_size
      end = self.index_in_epoch
    # return inputs and targets
    return [self.list_of_data[i][start:end] for i in range(self.num_data)]