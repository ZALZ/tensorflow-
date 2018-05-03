#coding:utf-8
import tensorflow as tf
import forward
import backward
import numpy as np
from PIL import Image

def restore_model(testPicArr):
    #创建一个默认图，在该图中执行以下操作
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32,[None,forward.INPUT_NODE])
        y = forward.forward(x,None)
        preValue = tf.argmax(y,1) #得到概率最大预测值

        #实现滑动平均模型，参数MOVING_AVERAGE_DECAY用于控制模型更新速度，训练过程会对每一个变量维护一个影子变量，这个影子变量的初始值就是相应变量的初始值
        variable_averages = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                preValue = sess.run(preValue,feed_dict={x:testPicArr})
                return preValue
            else:
                print ("NO CHECKPOINT FILE FOUND.")
                return -1

def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28,28),Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    threshold = 50 #设定合适的值域
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if(im_arr[i][j]<threshold):
                im_arr[i][j] = 0
            else: im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1,784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr,1.0/255.0)

    return img_ready


def application():
    testNum = input("输入测试图片张数：")
    for i in range(testNum):
        testPic = raw_input("图片路径：")
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print "该图片识别为",preValue

def main():
    application()

if __name__ == "__main__":
    main()
