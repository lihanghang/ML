#!/usr/local/bin/python  
'''
基于flask实现智能推理系统的贷后企业的新闻情感分类模型的调用
by hanghangli 20190220
'''
# coding=utf-8  
 # 首先加载必用的库
import re
import jieba # 结巴分词
# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
from flask import Flask,request
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np

app = Flask(__name__)  
 # 使用gensim加载预训练中文分词embedding
cn_model = KeyedVectors.load_word2vec_format('./Chinese-Word-Vectors/sgns.zhihu.bigram', 
                                          binary=False)
@app.route('/Sentiment')  
def predict_sentiment():
    num_words = 50000
    news = request.args.get("text")
    # print(text)
    # 去标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", news)
    # 分词
    cut = jieba.cut(text)
    cut_list = [ i for i in cut ]
    # tokenize
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            cut_list[i] = 0
    # padding
    pad_sequences = tf.contrib.keras.preprocessing.sequence.pad_sequences
    tokens_pad = pad_sequences([cut_list], maxlen=2480,
                           padding='pre', truncating='pre')
    # 超出五万个词向量的词用0代替# 超出五万个词 
    tokens_pad[ tokens_pad>=num_words ] = 0 
    # 输入
    with tf.gfile.FastGFile('Text_Sentiment.pb','rb') as f:
        # 复制定义好的计算图到新的图中，先创建一个空的图.
        graph_def = tf.GraphDef()
        # 加载proto-buf中的模型
        graph_def.ParseFromString(f.read())
        # 最后复制pre-def图的到默认图中.

        _ = tf.import_graph_def(graph_def, name='')    
    with tf.Session() as sess:
        # 初始化所有变量
        init = tf.global_variables_initializer()
        sess.run(init)
        input_x = sess.graph.get_tensor_by_name('cnn/placeholders/inputs:0')
        op = sess.graph.get_tensor_by_name('cnn/output/predictions:0')
        ret = sess.run(op, feed_dict={input_x: tokens_pad})
        finalRes = ret[0][0]
        # 最后返回字符串格式
        return str(finalRes)
if __name__ == '__main__': 
    app.run(host='124.16.71.33',port=8600)  
