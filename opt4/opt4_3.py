#coding:UTF-8
#设损失函数loss=(w+1)^2， 令w初值为5.反向传播就是求最优w，即求最小loss对应的w值
import tensorflow as tf
#定义带优化参数w初值为5
w = tf.Variable(tf.constant(5,dtype=tf.float32))
#定义损失函数loss
loss = tf.square(w+1)
#定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(0.2).minimizer(loss)
#生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40
    for i in range(STEPS):
        sess.run(train_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print "after %s steps : w is %f."%(i,w_val,loss_val)
