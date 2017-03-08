from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import argparse
import sys
import random
import tensorflow as tf

import input_data

FLAGS = None

CONV1_WEIGHT_SHAPE = [5, 5, 1, 36]
CONV1_BIAS_SHAPE = [36]
CONV2_WEIGHT_SHAPE = [5, 5, 36, 36]
CONV2_BIAS_SHAPE = [36]
CONV3_WEIGHT_SHAPE = [5, 5, 36, 64]
CONV3_BIAS_SHAPE = [64]
LOCAL4_INPUT_SIZE = 5 * 5 * 64
LOCAL4_OUTPUT_SIZE = 256


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def inference(images, keep_prob):
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = weight_variable(CONV1_WEIGHT_SHAPE)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
        biases = bias_variable(CONV1_BIAS_SHAPE)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1
    with tf.name_scope('pool1'):
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = weight_variable(CONV2_WEIGHT_SHAPE)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='VALID')
        biases = bias_variable(CONV2_BIAS_SHAPE)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # pool2
    with tf.name_scope('pool2'):
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = weight_variable(CONV3_WEIGHT_SHAPE)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='VALID')
        biases = bias_variable(CONV3_BIAS_SHAPE)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)

    # pool3
    with tf.name_scope('pool3'):
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID', name='pool3')

    # local4
    with tf.variable_scope('local4') as scope:
        reshape = tf.reshape(pool3, [-1, LOCAL4_INPUT_SIZE])
        weights = weight_variable([LOCAL4_INPUT_SIZE, LOCAL4_OUTPUT_SIZE])
        biases = bias_variable([LOCAL4_OUTPUT_SIZE])
        local4 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # dropout
    with tf.name_scope('dropout'):
        dropped = tf.nn.dropout(local4, keep_prob=keep_prob)

    # linear layer
    with tf.variable_scope('softmax_linear') as scope:
        weights = weight_variable([LOCAL4_OUTPUT_SIZE, 4])
        biases = bias_variable([4])
        softmax_linear = tf.add(tf.matmul(dropped, weights), biases, name=scope.name)
    return softmax_linear


def loss(logits, labels):
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
    return cross_entropy_mean


def train(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy
