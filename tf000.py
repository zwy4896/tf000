from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import tensorflow as tf 

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    # Model function for CNN
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # convolutional layer 1
    conv1 = tf.layers.conv2d(inputs = input_layer,
                            filters = 32,
                            kernel_size = [5, 5],
                            padding="same",
                            activation=tf.nn.relu)
    
    # Pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides = 2)

    # convolutional layer 2 & pooling layer 2 
    conv2 = tf.layers.conv2d(inputs = pool1,
    filters = 64,
    kernel_size = [5, 5],
    padding="same",
    activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides = 2)
if __name__ == "__main__":
    tf.app.run()
    