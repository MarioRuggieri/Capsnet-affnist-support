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

"""Tests for affnist_input_record."""

import numpy as np
import tensorflow as tf
import skimage.io as skio

import affnist_input_record

AFFNIST_DATA_DIR = '../../testdata/affnist/'

class AffnistInputRecordTest(tf.test.TestCase):

  def testTrain(self):
    with self.test_session(graph=tf.Graph()) as sess:
      new_height = 28
      features = affnist_input_record.inputs(
          data_dir=AFFNIST_DATA_DIR,
          batch_size=1,
          split='train',
          new_height=new_height,
          batch_capacity=2)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      images, labels, recons_image = sess.run([features['images'], features['labels'], features['recons_image']])
      if new_height is None:
          image_dim = 40
      else:
          image_dim = new_height
      self.assertEqual((1, 10), labels.shape)
      self.assertEqual(1, np.sum(labels))
      self.assertItemsEqual([0, 1], np.unique(labels))
      self.assertEqual(image_dim, features['height'])
      self.assertEqual((1, image_dim, image_dim, 1), images.shape)
      self.assertEqual(recons_image.shape, images.shape)
      self.assertAllEqual(recons_image, images)

      skio.imsave("decoded.jpg",images[0].squeeze())

      coord.request_stop()
      for thread in threads:
        thread.join()

if __name__ == '__main__':
  tf.test.main()
