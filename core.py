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


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def inference(images, keep_prob):
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = weight_variable([7, 7, 1, 36])
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
        biases = bias_variable([36])
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1
    with tf.name_scope('pool1'):
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = weight_variable([6, 6, 36, 36])
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='VALID')
        biases = bias_variable([36])
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # pool2
    with tf.name_scope('pool2'):
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = weight_variable([5, 5, 36, 64])
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='VALID')
        biases = bias_variable([64])
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)

    # pool3
    with tf.name_scope('pool3'):
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID', name='pool3')

    # local4
    with tf.variable_scope('local4') as scope:
        reshape = tf.reshape(pool3, [-1, 8 * 8 * 64])
        weights = weight_variable([8 * 8 * 64, 128])
        biases = bias_variable([128])
        local4 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # dropout
    with tf.name_scope('dropout'):
        dropped = tf.nn.dropout(local4, keep_prob=keep_prob)

    # linear layer
    with tf.variable_scope('softmax_linear') as scope:
        weights = weight_variable([128, 4])
        biases = bias_variable([4])
        return tf.add(tf.matmul(dropped, weights), biases, name=scope.name)


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


def train(loss, learning_rate):
    # sess = tf.InteractiveSession()
    #
    # with tf.name_scope('input'):
    #     x = tf.placeholder(tf.float32, [None, 128 * 128], name='x-input')
    #     y_ = tf.placeholder(tf.float32, [None, 4], name='y-input')
    #
    # with tf.name_scope('input_reshape'):
    #     image_shaped_input = tf.reshape(x, [-1, 128, 128, 1])
    #     tf.summary.image('input', image_shaped_input, 10)
    #
    # with tf.name_scope('dropout_keep_prob'):
    #     keep_prob = tf.placeholder(tf.float32)

    # conv1
    # with tf.variable_scope('conv1') as scope:
    #     kernel = weight_variable([7, 7, 1, 36])
    #     conv = tf.nn.conv2d(image_shaped_input, kernel, [1, 1, 1, 1], padding='VALID')
    #     biases = bias_variable([36])
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv1 = tf.nn.relu(pre_activation, name=scope.name)
    #
    # # pool1
    # with tf.variable_scope('pool1') as scope:
    #     pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=scope.name)
    #
    # # conv2
    # with tf.variable_scope('conv2') as scope:
    #     kernel = weight_variable([6, 6, 36, 36])
    #     conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='VALID')
    #     biases = bias_variable([36])
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv2 = tf.nn.relu(pre_activation, name=scope.name)
    #
    # # pool2
    # with tf.variable_scope('pool2') as scope:
    #     pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=scope.name)
    #
    # # conv3
    # with tf.variable_scope('conv3') as scope:
    #     kernel = weight_variable([5, 5, 36, 64])
    #     conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='VALID')
    #     biases = bias_variable([64])
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv3 = tf.nn.relu(pre_activation, name=scope.name)
    #
    # # pool3
    # with tf.variable_scope('pool3') as scope:
    #     pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID', name=scope.name)
    #
    # # local4
    # with tf.variable_scope('local4') as scope:
    #     reshape = tf.reshape(pool3, [-1, 8 * 8 * 64])
    #     weights = weight_variable([8 * 8 * 64, 128])
    #     biases = bias_variable([128])
    #     local4 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    #
    # # dropout
    # with tf.name_scope('dropout'):
    #     keep_prob = tf.placeholder(tf.float32)
    #     dropped = tf.nn.dropout(local4, keep_prob)
    #
    # # linear layer
    # with tf.variable_scope('softmax_linear') as scope:
    #     weights = weight_variable([128, 4])
    #     biases = bias_variable([4])
    #     y = tf.add(tf.matmul(dropped, weights), biases, name=scope.name)

    # with tf.name_scope('cross_entropy'):
    #     diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    #     with tf.name_scope('total'):
    #         cross_entropy = tf.reduce_mean(diff)
    # tf.summary.scalar('cross_entropy', cross_entropy)
    #
    # with tf.name_scope('train'):
    #     train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    #
    # with tf.name_scope('accuracy'):
    #     with tf.name_scope('correct_prediction'):
    #         correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #     with tf.name_scope('accuracy'):
    #         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # tf.summary.scalar('accuracy', accuracy)
    #
    # merged = tf.summary.merge_all()
    # train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    # test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    # tf.global_variables_initializer().run()
    #
    # for i in range(20):
    #     images, labels = input_data.get_train_data()
    #     image_num, _ = images.shape
    #     batch_num = int(image_num / 100)
    #     random_index = random.sample(range(batch_num), batch_num)
    #     for j in range(batch_num):
    #         step = i * batch_num + j
    #         index = random_index[j]
    #         xs = images[index * 100: (index + 1) * 100]
    #         ys = labels[index * 100: (index + 1) * 100]
    #         if step % 20 == 0:
    #             summary, acc = sess.run([merged, accuracy], feed_dict={x: xs, y_: ys, keep_prob: 1.0})
    #             test_writer.add_summary(summary, step)
    #             print("step %d, train accuracy %g" % (step, acc))
    #         else:
    #             summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y_: ys, keep_prob: FLAGS.dropout})
    #             train_writer.add_summary(summary, step)
    #
    # saver = tf.train.Saver()
    # checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
    # saver.save(sess, checkpoint_path)
    #
    # train_writer.close()
    # test_writer.close()

    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy
