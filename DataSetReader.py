import os
import json
import tensorflow as tf
import numpy as np
from collections import defaultdict
import time
import cv2

class Reader:
    def __init__(self, tfrecords_dir, tfrecords_name, label_dir, label_name, class_num, image_size):
        '''
        :param tfrecords_dir: 存储tfrecords文件的文件夹目录
        :param tfrecords_name: 存储tfrecords文件的文件夹名称
        :param label_dir:
        :param label_name:
        :param class_num: 有多少个类别
        :param image_size: 图像的尺寸， 例如[224, 224]
        '''

        self.label_dir = label_dir
        self.label_name = label_name
        self.tfrecords_dir = tfrecords_dir    # model_data
        self.tfrecords_name = tfrecords_name
        file_pattern = os.path.join(self.tfrecords_dir, self.tfrecords_name) + "/*" + '.tfrecords'
        self.TfrecordFile = tf.gfile.Glob(file_pattern)
        self.class_num = class_num
        self.image_size = image_size


    def parser(self, serialized_example):
        """
        Introduction
        ------------
            解析tfRecord数据
        Parameters
        ----------
            serialized_example: 序列化的每条数据
        """
        features = tf.parse_single_example(
            serialized_example,
            features = {
                'image/encoded' : tf.FixedLenFeature([], dtype = tf.string),
                'image/label' : tf.VarLenFeature(dtype = tf.int64)
            }
        )
        image = tf.image.decode_png(features['image/encoded'], channels = 3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32) * 2.0 - 1.0
        label = tf.one_hot(features['image/label'].values, self.class_num)
        image = tf.image.resize_images(image, self.image_size)
        return image, label




    def build_dataset(self, batch_size, epoch, shuffle_num=10000, is_train=True):
        """
        Introduction
        ------------
            建立数据集dataset
        Parameters
        ----------
            batch_size: batch大小
        Return
        ------
            dataset: 返回tensorflow的dataset
        """
        names = np.loadtxt(os.path.join(self.label_dir, self.label_name), dtype=np.str)
        dataset_size = names.shape[0]

        dataset = tf.data.TFRecordDataset(filenames = self.TfrecordFile)
        dataset = dataset.map(self.parser, num_parallel_calls = 10)
        if is_train:
            dataset = dataset.shuffle(shuffle_num).batch(batch_size).repeat(epoch)
        else:
            dataset = dataset.batch(batch_size).repeat(epoch)
        # dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        one_element = iterator.get_next()
        return one_element, dataset_size

if __name__ == '__main__':
    tfrecords_dir = '/home/yang/study/datasetandparam/parachute'
    tfrecords_name = 'cifar-100_tfrecords'
    label_dir = '/home/yang/study/datasetandparam/parachute'
    label_name = 'my_cifar-100.txt'
    class_num = 101
    image_size = (224, 224)
    dataset = Reader(tfrecords_dir, tfrecords_name, label_dir, label_name, class_num, image_size)
    one_element, dataset_size = dataset.build_dataset(batch_size=10, epoch=1, shuffle_num=1000, is_train=True)

    with tf.Session() as sess:
        batch_images, batch_labels = sess.run(one_element)
        batch_labels = np.reshape(batch_labels, (10, class_num))
        # print(batch_images, batch_labels)
        count = 0
        for image, label in zip(batch_images, batch_labels):
            count += 1
            image = ((image + 1) * 127.5).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
            print(label)
            cv2.imshow(str(count), image)
        cv2.waitKey(0)

