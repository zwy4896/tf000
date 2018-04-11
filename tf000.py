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

    # dense layer
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    dense = tf.layers.dense(inputs = pool2_flat, units = 1024, activiation = tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs = dense, rate = 0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # logits layer
    logits = tf.layers.dense(inputs=dropout, units = 10)

    predictions = {"class":tf.arg_max(input=logits, axis = 1),
    "probabilities":tf.nn.softmax(logits, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {"accuracy":tf.metrics.accuracy(labels=labels, predictions=predictions["class"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
if __name__ == "__main__":
    tf.app.run()
    