import random
import numpy
from PIL import Image, ImageFilter, ImageEnhance
import scipy.io

IMAGE_SIZE = 96
CROP_SIZE = IMAGE_SIZE / 2


class DataSet(object):
    def __init__(self, image_path, mask_path, start_index, end_index):
        self.ori_img = Image.open(image_path)
        self.mask = scipy.io.loadmat(mask_path)['mask']
        self.start_index = start_index
        self.end_index = end_index
        random_indexes = [i for i in range(end_index - start_index)]
        for index in range(start_index, end_index):
            random_indexes[index - start_index] = random.sample(range(0, 2000), 2000)
        self.random_indexes = random_indexes
        self.index = 0

    def get_data(self, index):
        images = numpy.zeros((0, IMAGE_SIZE * IMAGE_SIZE))
        labels = numpy.zeros((0, 4))
        ori_img = self.ori_img
        mask = self.mask
        start_index = self.start_index
        end_index = self.end_index
        for i in range(start_index, end_index):
            ori_img.seek(i)
            # enhancer = ImageEnhance.Contrast(ori_img)
            # x_img = enhancer.enhance(1.6)
            x_img = ori_img.filter(ImageFilter.SHARPEN)
            seg_slice = mask[:, :, i]
            random_index = self.random_indexes[i - start_index][index * 100:(index + 1) * 100]

            batch_x = numpy.zeros((0, IMAGE_SIZE, IMAGE_SIZE))
            batch_y = numpy.zeros((0, 4))

            # fibroglandular
            (f_x, f_y) = numpy.where(seg_slice == 255)
            f_num = f_x.size
            if f_num >= 2000:
                batch_f_x = numpy.zeros((100, IMAGE_SIZE, IMAGE_SIZE))
                batch_f_y = numpy.zeros((100, 4))
                f_random_x = f_x[random_index]
                f_random_y = f_y[random_index]
                for j in range(0, 100):
                    x = f_random_x[j]
                    y = f_random_y[j]
                    box = (y - CROP_SIZE, x - CROP_SIZE, y + CROP_SIZE, x + CROP_SIZE)
                    batch_f_x[j, :, :] = numpy.array(x_img.crop(box))
                    batch_f_y[j, 0] = 1
                batch_x = numpy.concatenate((batch_x, batch_f_x))
                batch_y = numpy.concatenate((batch_y, batch_f_y))

            # mass
            (m_x, m_y) = numpy.where(seg_slice == 100)
            m_num = m_x.size
            if m_num >= 2000:
                batch_m_x = numpy.zeros((100, IMAGE_SIZE, IMAGE_SIZE))
                batch_m_y = numpy.zeros((100, 4))
                m_random_x = m_x[random_index]
                m_random_y = m_y[random_index]
                for j in range(0, 100):
                    x = m_random_x[j]
                    y = m_random_y[j]
                    box = (y - CROP_SIZE, x - CROP_SIZE, y + CROP_SIZE, x + CROP_SIZE)
                    batch_m_x[j, :, :] = numpy.array(x_img.crop(box))
                    batch_m_y[j, 1] = 1
                batch_x = numpy.concatenate((batch_x, batch_m_x))
                batch_y = numpy.concatenate((batch_y, batch_m_y))

            # skin
            (s_x, s_y) = numpy.where(seg_slice == 150)
            s_num = s_x.size
            if s_num >= 2000:
                batch_s_x = numpy.zeros((100, IMAGE_SIZE, IMAGE_SIZE))
                batch_s_y = numpy.zeros((100, 4))
                s_random_x = s_x[random_index]
                s_random_y = s_y[random_index]
                for j in range(0, 100):
                    x = s_random_x[j]
                    y = s_random_y[j]
                    box = (y - CROP_SIZE, x - CROP_SIZE, y + CROP_SIZE, x + CROP_SIZE)
                    batch_s_x[j, :, :] = numpy.array(x_img.crop(box))
                    batch_s_y[j, 2] = 1
                batch_x = numpy.concatenate((batch_x, batch_s_x))
                batch_y = numpy.concatenate((batch_y, batch_s_y))

            # nothing
            (n_x, n_y) = numpy.where(seg_slice == 0)
            n_num = n_x.size
            if n_num >= 2000:
                batch_n_x = numpy.zeros((100, IMAGE_SIZE, IMAGE_SIZE))
                batch_n_y = numpy.zeros((100, 4))
                n_random_x = n_x[random_index]
                n_random_y = n_y[random_index]
                for j in range(0, 100):
                    x = n_random_x[j]
                    y = n_random_y[j]
                    box = (y - CROP_SIZE, x - CROP_SIZE, y + CROP_SIZE, x + CROP_SIZE)
                    batch_n_x[j, :, :] = numpy.array(x_img.crop(box))
                    batch_n_y[j, 3] = 1
                batch_x = numpy.concatenate((batch_x, batch_n_x))
                batch_y = numpy.concatenate((batch_y, batch_n_y))

            (n, w, h) = batch_x.shape
            batch_x = numpy.reshape(batch_x, (n, w * h))
            batch = numpy.concatenate((batch_x, batch_y), axis=1)
            numpy.random.shuffle(batch)
            batch_x, batch_y = numpy.split(batch, [IMAGE_SIZE * IMAGE_SIZE], axis=1)

            images = numpy.concatenate((images, batch_x))
            labels = numpy.concatenate((labels, batch_y))
        return images, labels


def get_train_data(index):
    (images_005, labels_005) = DataSet('E:\MachineLearning\segmentation\case_005.tif',
                                       'E:\MachineLearning\segmentation\mask_005.mat', 83, 106).get_data(index)
    (images_006, labels_006) = DataSet('E:\MachineLearning\segmentation\case_006.tif',
                                       'E:\MachineLearning\segmentation\mask_006.mat', 99, 120).get_data(index)
    (images_027, labels_027) = DataSet('E:\MachineLearning\segmentation\case_027.tif',
                                       'E:\MachineLearning\segmentation\mask_027.mat', 99, 109).get_data(index)
    (images_028, labels_028) = DataSet('E:\MachineLearning\segmentation\case_028.tif',
                                       'E:\MachineLearning\segmentation\mask_028.mat', 57, 61).get_data(index)
    images = numpy.concatenate((images_005, images_006, images_027, images_028), axis=0)
    labels = numpy.concatenate((labels_005, labels_006, labels_027, labels_028), axis=0)
    images = images.astype(numpy.float32)
    images = numpy.multiply(images, 1.0 / 255.0)
    labels = labels.astype(numpy.float32)
    return images, labels


def get_test_data():
    (images, labels) = DataSet('E:\MachineLearning\segmentation\case_005.tif',
                               'E:\MachineLearning\segmentation\mask_005.mat', 80, 81).get_data(0)
    images = images.astype(numpy.float32)
    images = numpy.multiply(images, 1.0 / 255.0)
    labels = labels.astype(numpy.float32)
    return images, labels
