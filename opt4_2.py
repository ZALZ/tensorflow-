#coding:UTF-8
#酸奶成本1元，利润9元。预测少了损失大，故生成的模型会多预测一些
#0导入模块，生成数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455
COST = 1
PROFIT = 9

rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
Y = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1,x2) in X]

#1定义神经网络的输入，参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w1)

#2定义损失函数及反向传播方法
#定义的损失函数是的预测少了损失大，故模型应该往偏多预测
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*COST,(y_-y)*PROFIT))
train_step = tf.train.GradientOptimizer(0.001).minimize(loss)
