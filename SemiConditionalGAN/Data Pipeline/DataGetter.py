import sys, os
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.preprocessing import OneHotEncoder
from imgaug import augmenters as iaa
from CustomDataSet import DataSet

seq = iaa.Sequential([
  iaa.Sometimes(0.1, iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},)),
  iaa.Sometimes(0.1, iaa.Affine(rotate=(-25, 25),)),
  iaa.Sometimes(0.1, iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},)),
  iaa.Fliplr(1.0),
  ], random_order=True)

def build_paired_data():
  onehot_encoder = OneHotEncoder(sparse=True)
  (train_batch, train_label), (test_batch, test_label) = tf.keras.datasets.mnist.load_data()
  train_batch, test_batch = np.reshape(train_batch, [60000, 28, 28, 1]), np.reshape(test_batch, [10000, 28, 28, 1])
  image_shape = [28, 28, 1]

  train_batch, test_batch = train_batch/255, test_batch/255

  train_filter = np.where(train_label == 0)
  train_data = train_batch[train_filter]
  perm = [n for n in range(train_data.shape[0])]
  np.random.shuffle(perm)
  train_data_pair = train_data[perm]
  train_labels = onehot_encoder.fit_transform(train_label[train_filter]).toarray().astype(np.float32)

  test_filter = np.where(test_label == 0)
  test_data = test_batch[test_filter]
  perm = [n for n in range(test_data.shape[0])]
  np.random.shuffle(perm)
  test_data_pair = test_data[perm]
  test_labels = onehot_encoder.fit_transform(test_label[test_filter]).toarray().astype(np.float32)

  train_labels = np.reshape(train_labels, [train_labels.shape[1], 1])
  test_labels = np.reshape(test_labels, [test_labels.shape[1], 1])

  for n in range(1, 10):
    train_filter = np.where(train_label == n)
    test_filter = np.where(test_label == n)    
    
    cspec_train_batch = train_batch[train_filter]
    perm = [n for n in range(cspec_train_batch.shape[0])]
    np.random.shuffle(perm)
    cspec_train_batch_pair = cspec_train_batch[perm]
    cspec_train_labels = onehot_encoder.fit_transform(train_label[train_filter]).toarray().astype(np.float32)
    train_data = np.vstack((train_data, cspec_train_batch))
    train_data_pair = np.vstack((train_data_pair, cspec_train_batch_pair))
    cspec_train_labels = np.reshape(cspec_train_labels, [cspec_train_labels.shape[1], 1])
    train_labels = np.vstack((train_labels, cspec_train_labels))

    cspec_test_batch = test_batch[test_filter]
    perm = [n for n in range(cspec_test_batch.shape[0])]
    np.random.shuffle(perm)
    cspec_test_batch_pair = cspec_test_batch[perm]
    cspec_test_labels = onehot_encoder.fit_transform(test_label[test_filter]).toarray().astype(np.float32)
    test_data = np.vstack((test_data, cspec_test_batch))
    test_data_pair = np.vstack((test_data_pair, cspec_test_batch_pair))
    cspec_test_labels = np.reshape(cspec_test_labels, [cspec_test_labels.shape[1], 1])
    test_labels = np.vstack((test_labels, cspec_test_labels))
  
  perm = [n for n in range(60000)]
  np.random.shuffle(perm)
  train_data, train_data_pair, train_labels = train_data[perm], train_data_pair[perm], train_labels[perm]

  perm = [n for n in range(10000)]
  np.random.shuffle(perm)
  test_data, test_data_pair, test_labels = test_data[perm], test_data_pair[perm], test_labels[perm]

  DataSet_train = DataSet([train_data, train_data_pair, train_labels])
  DataSet_test = DataSet([test_data, test_data_pair, test_labels])

  return DataSet_train, DataSet_test, image_shape


def get_data(data_name, use_aug=True):
  onehot_encoder = OneHotEncoder(sparse=True)
  if data_name == "cifar":
    (train_batch, train_label), (test_batch, test_label) = tf.keras.datasets.cifar10.load_data()
    image_shape = [32, 32, 3]
  elif data_name == "mnist":
    (train_batch, train_label), (test_batch, test_label) = tf.keras.datasets.mnist.load_data()
    train_batch, test_batch = np.reshape(train_batch, [60000, 28, 28, 1]), np.reshape(test_batch, [10000, 28, 28, 1])
    train_label, test_label = np.reshape(train_label, [60000, 1]), np.reshape(test_label, [10000, 1])
    image_shape = [28, 28, 1]
  elif data_name == "mnist_paired":
    return build_paired_data()
  elif data_name == "latent":
    DataSet_train, DataSet_test = H5DataSet(addresses="*train"), H5DataSet(addresses="*test")
    return DataSet_train, DataSet_test, [28, 28, 1]
  else:
    raise ValueError("data_name must be cifar, mnist, or latent.")
  train_label = onehot_encoder.fit_transform(train_label).toarray().astype(np.float32)
  test_label = onehot_encoder.fit_transform(test_label).toarray().astype(np.float32)  
  train_batch, test_batch = train_batch/255, test_batch/255
  if use_aug:
    DataSet_train = DataSet(train_batch, train_label, seq.augment_images)
  else:
    DataSet_train = DataSet([train_batch, train_label])
  DataSet_test = DataSet([test_batch, test_label])  
  return DataSet_train, DataSet_test, image_shape