from PIL import Image
import numpy
import random
import scipy.io


def get_data_from_case(image_path, mask_path, start_index, end_index):
    ori_img = Image.open(image_path)
    mask = scipy.io.loadmat(mask_path)['mask']

    images = numpy.zeros((0, 128 * 128))
    labels = numpy.zeros((0, 4))

    for i in range(start_index, end_index):

        ori_img.seek(i)
        seg_slice = mask[:, :, i]

        batch_x = numpy.zeros((0, 128, 128))
        batch_y = numpy.zeros((0, 4))

        # fibroglandular
        (f_x, f_y) = numpy.where(seg_slice == 255)
        f_num = f_x.size
        if f_num >= 1000:
            batch_f_x = numpy.zeros((100, 128, 128))
            batch_f_y = numpy.zeros((100, 4))
            f_random_index = random.sample(range(0, f_num), 100)
            f_random_x = f_x[f_random_index]
            f_random_y = f_y[f_random_index]
            for j in range(0, 100):
                x = f_random_x[j]
                y = f_random_y[j]
                box = (y - 64, x - 64, y + 64, x + 64)
                batch_f_x[j, :, :] = numpy.array(ori_img.crop(box))
                batch_f_y[j, 0] = 1
            batch_x = numpy.concatenate((batch_x, batch_f_x))
            batch_y = numpy.concatenate((batch_y, batch_f_y))

        # mass
        (m_x, m_y) = numpy.where(seg_slice == 100)
        m_num = m_x.size
        if m_num >= 1000:
            batch_m_x = numpy.zeros((100, 128, 128))
            batch_m_y = numpy.zeros((100, 4))
            m_random_index = random.sample(range(0, m_num), 100)
            m_random_x = m_x[m_random_index]
            m_random_y = m_y[m_random_index]
            for j in range(0, 100):
                x = m_random_x[j]
                y = m_random_y[j]
                box = (y - 64, x - 64, y + 64, x + 64)
                batch_m_x[j, :, :] = numpy.array(ori_img.crop(box))
                batch_m_y[j, 1] = 1
            batch_x = numpy.concatenate((batch_x, batch_m_x))
            batch_y = numpy.concatenate((batch_y, batch_m_y))

        # skin
        (s_x, s_y) = numpy.where(seg_slice == 150)
        s_num = s_x.size
        if s_num >= 1000:
            batch_s_x = numpy.zeros((100, 128, 128))
            batch_s_y = numpy.zeros((100, 4))
            s_random_index = random.sample(range(0, s_num), 100)
            s_random_x = s_x[s_random_index]
            s_random_y = s_y[s_random_index]
            for j in range(0, 100):
                x = s_random_x[j]
                y = s_random_y[j]
                box = (y - 64, x - 64, y + 64, x + 64)
                batch_s_x[j, :, :] = numpy.array(ori_img.crop(box))
                batch_s_y[j, 2] = 1
            batch_x = numpy.concatenate((batch_x, batch_s_x))
            batch_y = numpy.concatenate((batch_y, batch_s_y))

        # nothing
        (n_x, n_y) = numpy.where(seg_slice == 0)
        n_num = n_x.size
        if n_num >= 1000:
            batch_n_x = numpy.zeros((100, 128, 128))
            batch_n_y = numpy.zeros((100, 4))
            n_random_index = random.sample(range(0, n_num), 100)
            n_random_x = n_x[n_random_index]
            n_random_y = n_y[n_random_index]
            for j in range(0, 100):
                x = n_random_x[j]
                y = n_random_y[j]
                box = (y - 64, x - 64, y + 64, x + 64)
                batch_n_x[j, :, :] = numpy.array(ori_img.crop(box))
                batch_n_y[j, 3] = 1
            batch_x = numpy.concatenate((batch_x, batch_n_x))
            batch_y = numpy.concatenate((batch_y, batch_n_y))

        (n, w, h) = batch_x.shape
        batch_x = numpy.reshape(batch_x, (n, w * h))
        batch = numpy.concatenate((batch_x, batch_y), axis=1)
        numpy.random.shuffle(batch)
        batch_x, batch_y = numpy.split(batch, [128 * 128], axis=1)

        images = numpy.concatenate((images, batch_x))
        labels = numpy.concatenate((labels, batch_y))

    return images, labels


def get_train_data():
    (images_005, labels_005) = get_data_from_case('E:\MachineLearning\segmentation\case_005.tif',
                                                  'E:\MachineLearning\segmentation\mask_005.mat', 80, 105)
    (images_027, labels_027) = get_data_from_case('E:\MachineLearning\segmentation\case_027.tif',
                                                  'E:\MachineLearning\segmentation\mask_027.mat', 99, 109)
    (images_028, labels_028) = get_data_from_case('E:\MachineLearning\segmentation\case_028.tif',
                                                  'E:\MachineLearning\segmentation\mask_028.mat', 57, 61)
    images = numpy.concatenate((images_005, images_027, images_028), axis=0)
    labels = numpy.concatenate((labels_005, labels_027, labels_028), axis=0)
    images = images.astype(numpy.float32)
    images = numpy.multiply(images, 1.0 / 255.0)
    labels = labels.astype(numpy.float32)
    return images, labels


def get_test_data():
    (images, labels) = get_data_from_case('E:\MachineLearning\segmentation\case_005.tif',
                                          'E:\MachineLearning\segmentation\mask_005.mat', 80, 81)
    images = images.astype(numpy.float32)
    images = numpy.multiply(images, 1.0 / 255.0)
    labels = labels.astype(numpy.float32)
    return images, labels