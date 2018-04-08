#coding:UTF-8
#导入模块，生成数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE=8
seed=23445

#基于seed产生随机数,并返回32行2列的随机数矩阵，作为喂养数据集
rnp=np.random.RandomState(seed)
X=rnp.rand(32,2)

#从X中取出一行相加,如果小于1则给Y赋值1,否则复制0,作为数据集样本标签
Y=[[int(x0+x1<1)] for (x0,x1) in X]
//print "X:\n",X
//print "Y:\n",Y

#定义神经网络的输入、参数和输出，定义前向传播过程。
x =tf.placeholder(tf.float32,shape=(None,2))   
y_=tf.placeholder(tf.float32,shape=(None,1))

w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

#定义损失函数，及反向传播方法
loss=tf.reduce_mean(tf.square(y-y_))
train_step=tf.train.GradientDescentOptimizer(0.001).minimizer(loss)
#train_step=tf.train.MomentumOptimizer(0.001,0.9).minimizer(loss)
#train_step=tf.train.AdamOptimizer(0.001).minimizer(loss)

#生成会话,训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #输出当前未经训练的参数值
    print "w1:\n",sess.run(w1)
    print "w2:\n",sess.run(w2)
    print "\n"
    #训练模型
    STEPS=3000
    for i in range(STEPS):
        start=(i*BATCH_SIZE)%32
        end=start+BATCH_SIZE
        sess.run(train_step,feed_dict{x:X[start:end],y_:Y[start:end]})
        if i%500 == 0:
            total_loss=sess.run(loss,feed_dict{x:X,y_:Y})
            print "after %d rounds,loss on all data is %g"%(i,total_loss)
            
    #输出训练后的参数值
    print "w1:\n",sess.run(w1)
    print "w2:\n",sess.run(w2)
    print "\n"
