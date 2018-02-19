# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Input utility functions for mnist and mnist multi.

Handles reading from single digit, shifted single digit, and multi digit mnist
dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


def _read_and_decode(filename_queue, image_dim=40, new_dim=28):
  """Reads a single record and converts it to a tensor.

  Args:
    filename_queue: Tensor Queue, list of input files.
    image_dim: Scalar, the height (and width) of the image in pixels.
    new_dim: Scalar, the new height if you want to resize images.

  Returns:
    Dictionary of the (Image, label) and the image height.

  """
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64)
      })

  # Convert from a scalar string tensor (whose single string has
  # length image_pixel*image_pixel) to a uint8 tensor with shape
  # [image_pixel, image_pixel, 1].
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image = tf.reshape(image, [image_dim, image_dim, 1])
  image.set_shape([image_dim, image_dim, 1])

  if new_dim is not None:
    image = tf.image.resize_images(image,[new_dim,new_dim])
    image_dim = new_dim

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image = tf.cast(image, tf.float32) * (1. / 255)

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)
  features = {
      'images': image,
      'labels': tf.one_hot(label, 10),
      'recons_image': image,
      'recons_label': label,
  }
  return features,image_dim


def inputs(data_dir,
           batch_size,
           split,
           height=40,
           new_height=28,
           batch_capacity=5000):
  """Reads input data.

  Args:
    data_dir: Directory of the data.
    batch_size: Number of examples per returned batch.
    split: train or test
    height: image height.
    new_height: resized image height.
    batch_capacity: the number of elements to prefetch in a batch.

  Returns:
    Dictionary of Batched features and labels.

  """
  if split == "train" and "mnist" in data_dir:
      file = 'train_6shifted_mnist.tfrecords'
      filenames = [os.path.join(data_dir, file)]
  else:
      file_format = 'affnist_{}.tfrecords'
      filenames = [os.path.join(data_dir, file_format.format(split))]

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        filenames, shuffle=(split == 'train'))

    features,image_dim = _read_and_decode(filename_queue, image_dim=height, new_dim=new_height)
    if split == 'train':
      batched_features = tf.train.shuffle_batch(
          features,
          batch_size=batch_size,
          num_threads=2,
          capacity=batch_capacity + 3 * batch_size,
          # Ensures a minimum amount of shuffling of examples.
          min_after_dequeue=batch_capacity)
    else:
      batched_features = tf.train.batch(
          features,
          batch_size=batch_size,
          num_threads=1,
          capacity=batch_capacity + 3 * batch_size)
    batched_features['height'] = image_dim
    batched_features['depth'] = 1
    batched_features['num_targets'] = 1
    batched_features['num_classes'] = 10
    return batched_features