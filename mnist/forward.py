#coding:utf-8
#在前向传播过程中，需要定义网络模型输入层个数、隐藏层节点数、输出层个数，
#定义网络参数 w、偏置 b，定义由输入到输出的神经网络架构。 
#在前向传播过程中，规定网络输入结点为 784 个（代表每张输入图片的像素个数）， 隐藏层节点 500 个，输出节点 10 个（表示输出为数字 0-9
#的十分类） 。由输入层到隐藏层的参数 w1 形状为[784,500]，由隐藏层到输出层的参数 w2 形状为[500,10]，参数满足截断正态分布，并使用正则化，将每个参
#数的正则化损失加到总损失中。由输入层到隐藏层的偏置 b1 形状为长度为 500的一维数组，由隐藏层到输出层的偏置 b2 形状为长度为 10 的一维数组，初始化
#值为全 0。前向传播结构第一层为输入 x 与参数 w1 矩阵相乘加上偏置 b1，再经过 relu 函数，得到隐藏层输出 y1。前向传播结构第二层为隐藏层输出 y1 与参
#数 w2 矩阵相乘加上偏置 b2，得到输出 y。由于输出 y 要经过 softmax 函数，使其符合概率分布，故输出 y 不经过 relu 函数。 
import tensorflow as tf
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight(shape,regularizer):
    w= tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w
    
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def forward(x,regularizer):
    w1 = get_weight([INPUT_NODE,LAYER1_NODE],regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x,w1)+b1)
    
    w2 = get_weight([LAYER1_NODE,OUTPUT_NODE],regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.nn.relu(tf.matmul(y1,w2)+b2)
    return y
