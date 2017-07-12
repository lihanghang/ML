# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("tensorflow运行版本：" + tf.__version__)
mnist = input_data.read_data_sets('../MINST_DATA', one_hot=True)

chunk_size = 28
chunk_n = 28
rnn_size = 256
n_output_layer = 10

X = tf.placeholder('float', [None, chunk_n, chunk_size])
Y = tf.placeholder('float')

def recurrent_neural_network(data):
    layer = {'w_':tf.Variable(tf.random_normal([rnn_size, n_output_layer])), 'b_':tf.Variable(tf.random_normal([n_output_layer]))}
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)

    data = tf.transpose(data, [1, 0, 2])
    data= tf.reshape(data, [-1, chunk_size])
    data = tf.split(data, chunk_n, 0)
    # tf.nn.rnn->tf.nn.static_rnn
    outputs, status = tf.nn.static_rnn(lstm_cell, data, dtype=tf.float32)
    output = tf.add(tf.matmul(outputs[-1], layer['w_']), layer['b_'])
    return output

batch_size = 50

def train_neural_network(X, Y):
    predict = recurrent_neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=predict))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)

    epochs = 5
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        epoch_loss = 0
        for epoch in range(epochs):
            for i in range (int(mnist.train.num_examples/batch_size)):
                x, y = mnist.train.next_batch(batch_size)
                x = x.reshape([batch_size, chunk_n, chunk_size])
                _, c = sess.run([optimizer, cost_func], feed_dict={X:x, Y:y})
                epoch_loss += c
            print(epoch,':', epoch_loss)
        correct = tf.equal(tf.argmax(predict, 1),tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('准确率：', accuracy.eval({X:mnist.test.images.reshape(-1,chunk_n, chunk_size), Y:mnist.test.labels}))
train_neural_network(X, Y)