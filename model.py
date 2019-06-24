import tensorflow as tf
import numpy as np
import time, cv2, datetime, os
import networks
from DataSetReader import Reader
from scipy.spatial.distance import pdist

class SimilarityCompute:
    def __init__(self, sess, cfg):
        self.sess = sess
        self.cfg = cfg
        self.weight_decay = cfg.weight_decay
        self.num_classes = cfg.num_classes
        self.images = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size * self.cfg.num_gpus,
                                                              self.cfg.image_size[0], self.cfg.image_size[1], 3],
                                                name='input_image')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size * self.cfg.num_gpus, self.num_classes],
                                     name='label')
        self.ResNet18 = networks.ResNet18(self.cfg)
        self.is_train = self.cfg.is_train


        self.global_step = tf.Variable(0, name='global_step', trainable=True)

        # self.lr = tf.Variable(self.cfg.lr, name='lr')
        self.lr = self.cfg.lr
        self.op_t = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.0, beta2=0.99, epsilon=1e-8)

        # 读取数据集
        tensor_file_maker = Reader(self.cfg.tfrecords_dir, self.cfg.tfrecords_name, self.cfg.datalabel_dir,
                                   self.cfg.datalabel_name,
                                   self.num_classes, self.cfg.image_size)
        self.one_element, self.dataset_size = tensor_file_maker.build_dataset(batch_size=self.cfg.batch_size * self.cfg.num_gpus,
                                                                    epoch=self.cfg.epoch, shuffle_num=10000,
                                                                    is_train=self.cfg.is_train)

        self.num_step_epoch = self.dataset_size // (self.cfg.batch_size * self.cfg.num_gpus)


    def train(self):
        prelogits, embedding = self.inference(self.images)
        self.logits = networks.m4_linear(prelogits, self.num_classes, active_function=None, norm=None,
                                    get_vars_name=False, is_trainable=self.is_train,
                                    stddev=0.02, weight_decay=self.weight_decay, name='logits') # 总共多少类， 注：训练时不需要l2正则化

        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
        #                                                                logits=logits, name='cross_entropy') # 接softmax，交叉熵

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,
                                                                       logits=self.logits,
                                                                       name='cross_entropy')  # 接softmax，交叉熵

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) # 获取权重的regularization loss，让权重更小，防止过拟合
        self.total_loss = tf.add_n([cross_entropy_mean] + regularization_loss) # 只要在前向cross_entropy_mean损失加上regularization loss即可，
                                                                               # train的时候会自动减小权重

        loss_sum = tf.summary.scalar('loss', self.total_loss)
        self.lr = tf.train.exponential_decay(learning_rate=self.cfg.lr, global_step=self.global_step, decay_steps=5 * self.num_step_epoch,
                                       decay_rate=0.9, staircase=True)
        vars = tf.trainable_variables()

        grad = self.op_t.compute_gradients(loss=self.total_loss, var_list=vars)
        self.opt = self.op_t.apply_gradients(grads_and_vars=grad, global_step=self.global_step)



        # 定义保存模型
        try:
            self.saver = tf.train.Saver()
        except:
            print('one model save error....')

        # 初始化变量
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.writer = tf.summary.FileWriter('{}/{}'.format(self.cfg.log_dir,
                                                           time.strftime("%Y-%m-%d %H:%M:%S",
                                                                         time.localtime(time.time()))),
                                            self.sess.graph)
        merged = tf.summary.merge_all()
        # 载入模型
        could_load, counter = self.load(self.cfg.checkpoint_dir, self.cfg.dataset_name)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")



        self.sess.graph.finalize()
        for epoch in range(1, self.cfg.epoch + 1):
            for batch_step in range(1, self.num_step_epoch+1):
                start_time = datetime.datetime.now()
                batch_images, batch_labels = self.sess.run(self.one_element)
                if batch_images.shape[0] < self.cfg.batch_size:
                    print(batch_images.shape)
                    print(batch_labels.shape)
                    continue
                batch_labels = np.reshape(batch_labels, (self.cfg.batch_size, self.num_classes))

                _, loss, counter, lr, output = self.sess.run([self.opt, self.total_loss, self.global_step, self.lr, self.logits],
                                        feed_dict={self.images: batch_images,
                                                   self.labels: batch_labels})
                accuray_count = 0
                for index_output, index_label in zip(output ,batch_labels):
                    if index_output.argmax() == index_label.argmax():
                        accuray_count += 1
                accuray = accuray_count / self.cfg.batch_size

                if batch_step % self.cfg.add_summary_period == 0:
                    [merged_] = self.sess.run([merged], feed_dict={self.images: batch_images,
                                                   self.labels: batch_labels})
                    self.writer.add_summary(merged_, counter)
                    print('add summary once....')

                end_time = datetime.datetime.now()
                timediff = (end_time - start_time).total_seconds()
                print("Epoch: [%2d/%2d] [%4d/%4d], time: %3.4f, lr: %.6f accuray: %.4f Loss: %3.4f" %
                      (epoch, self.cfg.epoch, batch_step, self.num_step_epoch, timediff, lr, accuray, loss))

                if epoch % self.cfg.savemodel_period == 0 and batch_step == 1:
                    self.save(self.cfg.checkpoint_dir, epoch, self.cfg.dataset_name)
                    print('one param model saved....')


    def test(self):
        m4_temp = cv2.imread('/home/yang/plan2.jpeg')
        m4_img1 = cv2.imread('/home/yang/plan1.jpg')

        m4_temp = cv2.resize(m4_temp, (self.cfg.image_size[0], self.cfg.image_size[1]))
        m4_img1 = cv2.resize(m4_img1, (self.cfg.image_size[0], self.cfg.image_size[1]))
        m4_temp_f = m4_temp.astype(np.float32) / 127.5 - 1.0
        m4_img1_f = m4_img1.astype(np.float32) / 127.5 - 1.0

        img1_np = cv2.cvtColor(m4_temp_f, cv2.COLOR_BGR2RGB)
        img2_np = cv2.cvtColor(m4_img1_f, cv2.COLOR_BGR2RGB)

        images_list = [img1_np, img2_np]
        images_np = np.array(images_list)

        images = tf.placeholder(dtype=tf.float32, shape=[2, self.cfg.image_size[0], self.cfg.image_size[1], 3],
                                                name='input_image')
        prelogits, embedding = self.inference(images)

        # 定义保存模型
        try:
            self.saver = tf.train.Saver()
        except:
            print('one model save error....')

        # 初始化变量
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # 载入模型
        could_load, counter = self.load(self.cfg.checkpoint_dir, self.cfg.dataset_name)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for i in range(10):
            starttime = time.time()
            [m4_embedding] = self.sess.run([embedding], feed_dict={images: images_np})

            # print(m4_embedding[0])
            # print(m4_embedding[1])
            print(1 - pdist(m4_embedding, 'cosine'))
            endtime = time.time()
            print(endtime - starttime)


    def inference(self, image):
        with tf.variable_scope('similaritycompute610', reuse=False) as scope:
            prelogits = self.ResNet18.build_model(image)
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

    def save(self, checkpoint_dir, step, model_file_name):
        model_name = "GAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, model_file_name)

        # if not os.path.exists(checkpoint_dir):
        #     os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)
