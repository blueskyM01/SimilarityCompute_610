
# 分类网络的注意点
## 1. 分类的全连接层`不要加激活函数`, 同时， 分类的全连接层的前一层也`不要加激活函数`
## 2. weight decay的用法
* 2.1 在定义变量时加上L2正则化`l2_reg = tf.contrib.layers.l2_regularizer(weight_decay)`， 如下面代码所示
    ````
    l2_reg = tf.contrib.layers.l2_regularizer(weight_decay)
    batch, heigt, width, nc = input_.get_shape().as_list()
    w = tf.get_variable(name='filter', shape=[k_h, k_w, nc, fiters], regularizer=l2_reg,
    initializer=tf.truncated_normal_initializer(stddev=stddev))
    bias = tf.get_variable(name='biases', shape=[fiters], regularizer=l2_reg, initializer=tf.constant_initializer(0.0))
    ````
* 2.2 如何加上L2损失
  只需要在与label获得损失的基础上加上L2损失，如下面代码所示
    ````
    regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) # 获取权重的regularization loss，让权重更小，防止过拟合
    self.total_loss = tf.add_n([cross_entropy_mean] + regularization_loss) # 只要在前向cross_entropy_mean损失加上regularization loss即可，train的时候会自动减小权重
    ````
## 3.学习率
* 3.1 Reset18
   * lr=0.0001 开始下降
   * WD=0.0005
* 3.2 Reset34
   * lr=0.0001 开始下降
   * WD=0.0005
* 3.3 Reset50
   * lr=0.0001 开始下降
   * WD=0.0005
* 3.4 Reset101
   * lr=0.00001 开始下降, 最终学习率：0.000005（训练精度高95%以上，但测试精度只有48.3%）
   * WD=0.0005



