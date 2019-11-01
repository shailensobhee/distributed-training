# coding: utf-8
#==============================================================
#
# SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
# http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
#
# Copyright 2018 Intel Corporation
#
# THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
#
# =============================================================================
# ===============================================================
# Based on original examples of the TensorFlow repository 
#               from the TensorFlow Authors
# ===============================================================
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""

import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

FLAGS = None
import horovod.tensorflow as hvd
def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def cnn_model_fn(x):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(x, [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.5, training=True)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)
  return logits


#Helper function to parition mnist dataset based on ranks and size  
def get_dataset(rank, size):     
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data('MNIST-data-%d' % rank)
    x_train = x_train[rank::size]
    y_train = y_train[rank::size]
    x_test = x_test[rank::size]
    y_test = y_test[rank::size]
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    return (x_train, y_train), (x_test, y_test)

#Hepler function to randomize train dataset and create genrator object
def train_input_generator(x_train, y_train, batch_size=64):
    assert len(x_train) == len(y_train)
    while True:
        p = np.random.permutation(len(x_train))
        x_train, y_train = x_train[p], y_train[p]
        index = 0
        while index <= len(x_train) - batch_size:
            yield x_train[index:index + batch_size], \
                  y_train[index:index + batch_size],
            index += batch_size 

if __name__=="__main__":
  
    #initialize Horovod
    hvd.init()
    # Download and load MNIST dataset.
    # the downloaded dataset will be located at the default location ~/.keras/dataset
    # this cache location will be used once dataset is downloaded
    (x_train, y_train), (x_test, y_test) = get_dataset(hvd.rank(),hvd.size())    

    # The shape of downloaded data is (-1, 28, 28), hence we need to reshape it
    # into (-1, 784) to feed into our network. Also, need to normalize the
    # features between 0 and 1.
    x_train = np.reshape(x_train, (-1, 784)) / 255.0
    x_test = np.reshape(x_test, (-1, 784)) / 255.0

    # this will enable stdio logging
    tf.logging.set_verbosity(tf.logging.INFO)

    # Create plceholders for train data
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.int64, [None])

    # Build the graph for the deep net
    y_conv= cnn_model_fn(x)
    # define the loss
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    #multiply learning rate by #ranks due to the larger global batch size
    #Note that you are using horovod to compute avarge gradients
    opt = tf.train.AdamOptimizer(1e-4 * hvd.size())
    opt = hvd.DistributedOptimizer(opt)
    global_step = tf.train.get_or_create_global_step()
    train_step = opt.minimize(cross_entropy, global_step=global_step)

    # define accuracy - notice the difference to the loss function
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    #directory for MonitoredTrainingSession checkpoints - only rank 0
    checkpoint_dir = 'graphs/horovod' if hvd.rank() == 0 else None
    hooks = [
      #rank 0 will broadcast global variabes -> having equal weights on all nodes
      hvd.BroadcastGlobalVariablesHook(0),
      #dividing #steps by #ranks to address the increased learning rate
      tf.train.StopAtStepHook(last_step=1000 // hvd.size()),
      tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': cross_entropy, 'accuracy': accuracy}, every_n_iter=100),
    ]
    #Data genrator with batch_size=50 is used 
    training_batch_generator = train_input_generator(x_train,
                                                         y_train, batch_size=50)

    #Training starts                                                     
    time_start = time.time()
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir, hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        image_, label_ = next(training_batch_generator)
        mon_sess.run(train_step, feed_dict={x: image_, y_: label_}) 
    if hvd.rank() == 0:
      print(hvd.rank())
      print('TTT: %g' % (time.time() - time_start)) 

    #Use the checkpoint file to test the accuracy on test data   
    if hvd.rank() == 0:
      with tf.Session() as sess:
        tf_saver = tf.train.Saver()
        tf_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        acc = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
        print('test accuracy %g' % (acc))
