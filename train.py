import argparse
import os.path
import sys
import tensorflow as tf
import random
import time

import core
import input_data

FLAGS = None


def train():
    start_time = time.time()
    with tf.Graph().as_default():
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 128 * 128], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, 4], name='y-input')
        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(x, [-1, 128, 128, 1])
            tf.summary.image('input', image_shaped_input, 10)
        with tf.name_scope('dropout_keep_prob'):
            keep_prob = tf.placeholder(tf.float32)
        y = core.inference(image_shaped_input, keep_prob=keep_prob)
        loss = core.loss(y, y_)
        train_op = core.train(loss, FLAGS.learning_rate)
        accuracy = core.evaluation(y, y_)
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        sess.run(init)
        for i in range(20):
            images, labels = input_data.get_train_data()
            image_num, _ = images.shape
            batch_num = int(image_num / 100)
            random_index = random.sample(range(batch_num), batch_num)
            for j in range(batch_num):
                step = i * batch_num + j
                index = random_index[j]
                xs = images[index * 100: (index + 1) * 100]
                ys = labels[index * 100: (index + 1) * 100]
                if step % 20 == 0:
                    feed_dict = {x: xs, y_: ys, keep_prob: 1.0}
                    summary_str, acc = sess.run([summary, accuracy], feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    print("step %d, train accuracy %g" % (step, acc))
                else:
                    feed_dict = {x: xs, y_: ys, keep_prob: FLAGS.dropout_keep_prob}
                    summary_str, loss_value, _ = sess.run([summary, loss, train_op], feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                if (step + 1) % 1000 == 0:
                    checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
        duration = time.time() - start_time
        print('%d seconds' % int(duration))


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5,
                        help='Keep probability for training dropout.')
    parser.add_argument('--log_dir', type=str, default='E:\MachineLearning\segmentation\code\logs',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
