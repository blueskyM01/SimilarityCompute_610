from __future__ import division, print_function, absolute_import
import os
import argparse
import tensorflow as tf
from model import SimilarityCompute
import time



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
# TF_CPP_MIN_LOG_LEVEL 取值 0 ： 0也是默认值，输出所有信息
# TF_CPP_MIN_LOG_LEVEL 取值 1 ： 屏蔽通知信息
# TF_CPP_MIN_LOG_LEVEL 取值 2 ： 屏蔽通知信息和警告信息
# TF_CPP_MIN_LOG_LEVEL 取值 3 ： 屏蔽通知信息、警告信息和报错信息


parser = argparse.ArgumentParser()

# -----------------------------m4_BE_GAN_network-----------------------------
parser.add_argument("--gpu_assign", default='1', type=str, help="assign gpu")
parser.add_argument("--is_train", default=True, type=bool, help="Train")
parser.add_argument("--dataset_dir", default='/media/yang/F/DataSet/ImageNet', type=str, help="Train data set dir")
parser.add_argument("--dataset_name", default='image_train_jieya', type=str, help="Train data set name")
parser.add_argument("--datalabel_dir", default='/media/yang/F/DataSet/ImageNet', type=str, help="Train data label dir")
parser.add_argument("--datalabel_name", default='image_train_jieya.txt', type=str, help="Train data label name")
parser.add_argument("--tfrecords_dir", default='/media/yang/F/DataSet/ImageNet',
                    type=str, help="tfrecords_dir")
parser.add_argument("--tfrecords_name", default='tfrecords_imagenet',
                    type=str, help="tfrecords_name")
parser.add_argument("--log_dir", default='/media/yang/F/ubuntu/imagenet/muti-gpu/log', type=str, help="Train data label name")
parser.add_argument("--checkpoint_dir", default='/media/yang/F/ubuntu/imagenet/muti-gpu/checkpoint', type=str, help="model save dir")
parser.add_argument("--num_gpus", default=2, type=int, help="num of gpu")
parser.add_argument("--epoch", default=20, type=int, help="epoch")
parser.add_argument("--batch_size", default=128, type=int, help="batch size for one gpus")
parser.add_argument("--lr", default=0.00005, type=float, help="learning rate")
parser.add_argument("--savemodel_period", default=2, type=int, help="savemodel_period")
parser.add_argument("--add_summary_period", default=20, type=int, help="add_summary_period")
parser.add_argument("--weight_decay", default=0.0005, type=float, help="weight decay")
parser.add_argument("--num_classes", default=1001, type=int, help="num of classes")
parser.add_argument("--image_size", default=(224, 224), type=int, help="num of classes")

cfg = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_assign  # 指定第  块GPU可用

if __name__ == '__main__':

    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        m4_SimilarityCompute = SimilarityCompute(sess, cfg)
        if cfg.is_train:
            if not os.path.exists(cfg.log_dir):
                os.makedirs(cfg.log_dir)

            if not os.path.exists(cfg.checkpoint_dir):
                os.makedirs(cfg.checkpoint_dir)

            m4_SimilarityCompute.train()
        else:
            print('test starting ....')
            time.sleep(0.5)

            m4_SimilarityCompute.test()