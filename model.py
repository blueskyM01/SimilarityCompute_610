import tensorflow as tf
import numpy as np
import time, cv2
import networks
from DataSetReader import *

class SimilarityCompute:
    def __init__(self, sess, cfg):
        self.sess = sess
        self.cfg = cfg
        self.weight_decay = cfg.weight_decay
        self.num_classes = cfg.num_classes
        self.images = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size * self.cfg.num_gpus, 224, 224, 3],
                                     name='input_image')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size * self.cfg.num_gpus, self.num_classes],
                                     name='label')

        self.m4_Vgg16 = networks.Vgg16(self.cfg)
        self.is_train = self.cfg.is_train

        self.lr = tf.Variable(self.cfg.lr, name='lr')
        self.op_t = tf.train.AdamOptimizer(learning_rate=self.lr)


    def train(self):
        prelogits, embedding = self.inference(self.images)
        logits = networks.m4_linear(prelogits, self.num_classes, active_function=None, norm=None,
                                    get_vars_name=False, is_trainable=self.is_train,
                                    stddev=0.02, weight_decay=self.weight_decay, name='logits') # 总共多少类， 注：训练时不需要l2正则化

        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
        #                                                                logits=logits, name='cross_entropy') # 接softmax，交叉熵

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,
                                                                       logits=logits,
                                                                       name='cross_entropy')  # 接softmax，交叉熵

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) # 获取权重的regularization loss，让权重更小，防止过拟合
        self.total_loss = tf.add_n([cross_entropy_mean] + regularization_loss) # 只要在前向cross_entropy_mean损失加上regularization loss即可，
        self.opt = self.op_t.minimize(self.total_loss)                                                                  # train的时候会自动减小权重


        try:
            self.saver = tf.train.Saver()
        except:
            print('one model save error....')
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.writer = tf.summary.FileWriter('{}/{}'.format(self.cfg.log_dir,
                                                           time.strftime("%Y-%m-%d %H:%M:%S",
                                                                         time.localtime(time.time()))),
                                            self.sess.graph)

        could_load, counter = self.load(self.cfg.checkpoint_dir, self.cfg.dataset_name)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        tensor_file_maker = Reader(self.cfg.tfrecord_path, self.cfg.datalabel_dir, self.cfg.datalabel_name)
        one_element, dataset_size = tensor_file_maker.build_dataset(batch_size=self.cfg.batch_size * self.cfg.num_gpus,
                                                                    epoch=self.cfg.epoch, is_train=self.cfg.is_train)
        while 1:
            batch_images, batch_labels = self.sess.run(one_element)
            batch_labels = np.reshape(batch_labels, (self.cfg.batch_size,self.num_classes))
            _, loss = self.sess.run([self.opt, self.total_loss],
                                                   feed_dict={self.images: batch_images,
                                                              self.labels: batch_labels})
            print(loss)

    def test(self):
        print('a')

    def inference(self, image):
        with tf.variable_scope('similaritycompute610', reuse=False) as scope:
            prelogits = self.m4_Vgg16.build_model(image)
            embedding = tf.nn.l2_normalize(prelogits, 1, name='embedding') # 预测时l2正则化，训练时不需要
            return prelogits, embedding

    def load(self, checkpoint_dir, model_folder_name):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, model_folder_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            time.sleep(3)
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            time.sleep(3)
            return False, 0
