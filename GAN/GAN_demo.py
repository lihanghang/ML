# _*_coding:utf-8 _*_
import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib
matplotlib.use('nbagg')
import matplotlib.gridspec as gd
import pydevd



class GAN(object):

    def __init__(self, gen_shape, dis_shape, batch_size,step_num, learning_rate, data_width, data_length, data):
        self.gen_shape = gen_shape
        self.dis_shape = dis_shape
        self.batch_size = batch_size
        self.step_num = step_num
        self.learning_rate = learning_rate
        self.data_width = data_width
        self.data_length = data_length
        self.data = data
        self.gen_W = []
        self.gen_b = []
        self.dis_W = []
        self.dis_b = []
        self._creat_model()

    def _optimizer(self, loss, var_list):
        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss,var_list=var_list)
        return opt

    def _generator(self, z):
        current_input = z
        for i in range(len(self.gen_W)-1):
            current_output = tf.nn.relu(tf.matmul(current_input,self.gen_W[i]) + self.gen_b[i])
            current_input = current_output
        current_output = tf.nn.sigmoid(tf.matmul(current_input, self.gen_W[len(self.gen_W)-1]) + self.gen_b[len(self.gen_W)-1])
        return current_output

    def _discriminator(self, x):
        current_input = x
        for i in range(len(self.dis_W)-1):
            current_output = tf.nn.relu(tf.matmul(current_input,self.dis_W[i]) + self.dis_b[i])
            current_input = current_output
        current_output = tf.nn.sigmoid(tf.matmul(current_input, self.dis_W[len(self.dis_W)-1]) + self.dis_b[len(self.dis_W)-1])
        return current_output


    def _creat_vars(self):
        gen_len = len(gen_shape)
        dis_len = len(dis_shape)
        with tf.variable_scope('Generator'):
            for gen_i in range(gen_len - 1):
                stddev = 1. / tf.sqrt(gen_shape[gen_i] / 2.)
                W = tf.Variable(tf.random_normal([gen_shape[gen_i], gen_shape[gen_i+1]], stddev=stddev),
                                name="W"+str(gen_i))
                b = tf.Variable(tf.zeros([gen_shape[gen_i+1]]), name='b'+str(gen_i))
                self.gen_W.append(W)
                self.gen_b.append(b)
        with tf.variable_scope('Discriminator'):
            for dis_i in range(dis_len - 1):
                stddev = 1. / tf.sqrt(dis_shape[dis_i] / 2.)
                W = tf.Variable(tf.random_normal([dis_shape[dis_i], gen_shape[dis_i+1]], stddev=stddev),
                                name="W"+str(dis_i))
                b = tf.Variable(tf.zeros([gen_shape[dis_i+1]]), name='b'+str(dis_i))
                self.dis_W.append(W)
                self.dis_b.append(b)
    def _creat_model(self):
        self.z = tf.placeholder(tf.float32, [None, gen_shape[0]], name='z')
        self.x = tf.placeholder(tf.float32, [None, dis_shape[0]], name='x')
        self._creat_vars()

        self.g = self._generator(self.z)
        self.D_x = self._discriminator(self.x)
        self.D_g = self._discriminator(self.g)


    def _display(self):
        zs = np.random.uniform(-1., 1., size=[16, self.gen_shape[0]])
        gs = self.sess.run(self.g, feed_dict={self.z:zs})
        fig = plt.figure(figsize=(4, 4))
        graph = gd.GridSpec(4, 4)
        graph.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(gs):
            ax = plt.subplot(graph[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        return fig
    def train(self):
        loss_dis = -tf.reduce_mean(tf.log(self.D_x) + tf.log(1 - self.D_g))
        loss_gen = -tf.reduce_mean(tf.log(self.D_g))
        opt_dis = self._optimizer(loss_dis, var_list=self.dis_W+self.dis_b)
        opt_gen = self._optimizer(loss_gen, var_list=self.gen_W + self.gen_b)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        disp_step_num = 2000
        display_num = 10
        if not os.path.exists('out/'):
            os.makedirs('out/')
        fig_i = 0
        for step in range(self.step_num):
            xs, ys = self.data.train.next_batch(batch_size)
            zs = np.random.uniform(-1., 1., size=[batch_size, self.gen_shape[0]])
            _, l_dis = self.sess.run([opt_dis, loss_dis], feed_dict={self.z:zs, self.x:xs})
            _, l_gen = self.sess.run([opt_gen, loss_gen], feed_dict={self.z: zs})

            if step % 200 == 0:
                print(
                    'step: {}, loss_dis = {: .5}. loss_gen = {:.5}'.format(step, l_dis, l_gen)

                )
            if step % disp_step_num == 0:
                fig = self._display()
                plt.savefig('out/{}.png'.format(str(fig_i).zfill(3)), bbox_inches='tight')
                fig_i += 1
                plt.close(fig)
        self.sess.close()

if __name__ == '__main__':
    data = input_data.read_data_sets('../MINST_DATA', one_hot=True)
    learning_rate = 0.0002
    step_num = 30000
    batch_size = 32
    gen_shape = [10, 128, 784]
    dis_shape = [784, 128, 1]
    ae = GAN(
        gen_shape=gen_shape,
        dis_shape=dis_shape,
        batch_size=batch_size,
        step_num=step_num,
        learning_rate=learning_rate,
        data_width=28,
        data_length=28,
        data=data
    )
    ae.train()