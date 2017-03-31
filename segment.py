import numpy
import scipy.io
import tensorflow as tf
import os.path
import time
from PIL import Image

import core

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', 'E://MachineLearning/segmentation/code/logs',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('output_dir', 'E://MachineLearning/segmentation/code/out',
                           """Directory where to save segmented images.""")

IMAGE_SIZE = 96
CROP_SIZE = IMAGE_SIZE / 2


def segment_image(image_path, image_index):
    start_time = time.time()
    with tf.Session() as sess:
        x = tf.placeholder("float", [None, IMAGE_SIZE * IMAGE_SIZE])
        x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        y_conv = tf.nn.softmax(core.inference(x_image, keep_prob=1.0))
        prediction = tf.argmax(y_conv, 1)

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)

            original_img = Image.open(image_path)
            original_img.seek(image_index)
            (width, height) = original_img.size
            result = numpy.zeros((height, width))
            for i in range(0, height):
                for j in range(0, width):
                    box = (j - CROP_SIZE, i - CROP_SIZE, j + CROP_SIZE, i + CROP_SIZE)
                    image = numpy.array(original_img.crop(box))
                    image = numpy.reshape(image, (1, IMAGE_SIZE * IMAGE_SIZE))
                    image = image.astype(numpy.float32)
                    image = numpy.multiply(image, 1.0 / 255.0)
                    y_pred = prediction.eval(feed_dict={x: image})
                    print("%d, %d" % (i, j))
                    result[i][j] = y_pred
            filename = "%s_%d" % (image_path.split('\\')[-1].split('.')[0], image_index + 1)
            scipy.io.savemat(os.path.join(FLAGS.output_dir, filename), mdict={'result': result})
            duration = time.time() - start_time
            print('%d seconds' % int(duration))
        else:
            print('No checkpoint file found')
            return


def main(_):
    segment_image("E:\MachineLearning\segmentation\case_005.tif", 99)


if __name__ == '__main__':
    tf.app.run()
