# coding: utf-8


# 首先加载必用的库
import numpy as np
import matplotlib.pyplot as plt
import re
import jieba  # 结巴分词
# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
import warnings
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.framework import graph_util
warnings.filterwarnings("ignore")


# 获得正负向企业新闻数据
# pos数据
pos_txts = os.listdir('./CompanyNewsData/pos')
# neg数据
neg_txts = os.listdir('./CompanyNewsData/neg')
print('pos样本: ' + str(len(pos_txts)))
print('neg样本：' + str(len(neg_txts)))

# 使用gensim加训载预练中文分词embedding
cn_model = KeyedVectors.load_word2vec_format('./Chinese-Word-Vectors/sgns.zhihu.bigram',
                                            binary=False)
# 现在我们将所有的评价内容放置到一个list里
train_texts_orig = []  # 存储所有评价，每例评价为一条string
# 其中前7769条文本为正面评价，后7769条为负面评价
for i in range(len(pos_txts)):
    with open('./CompanyNewsData/pos/' + pos_txts[i], 'r', errors='ignore') as f:
        text = f.read().strip()
        train_texts_orig.append(text)
        f.close()
for i in range(len(pos_txts)):
    with open('./CompanyNewsData/neg/' + neg_txts[i], 'r', errors='ignore') as f:
        text = f.read().strip()
        train_texts_orig.append(text)
        f.close()

print(len(train_texts_orig))

# 进行分词和tokenize
# train_tokens是一个长长的list，其中含有4000个小list，对应每一条评价
train_tokens = []
for text in train_texts_orig:
    # 去掉标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
    # 结巴分词
    cut = jieba.cut(text)
    # 结巴分词的输出结果为一个生成器
    # 把生成器转换为list
    cut_list = [i for i in cut]
    for i, word in enumerate(cut_list):
        try:
            # 将词转换为索引index
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            # 如果词不在字典中，则输出0
            cut_list[i] = 0
    train_tokens.append(cut_list)

# 获得所有tokens的长度# 获得所有
num_tokens = [len(tokens) for tokens in train_tokens]
num_tokens = np.array(num_tokens)


# 取tokens平均值并加上两个tokens的标准差，
# 假设tokens长度的分布为正态分布，则max_tokens这个值可以涵盖95%左右的样本
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)

# 大约95%的样本被涵盖
# 我们对长度不足的进行padding，超长的进行修剪
np.sum(num_tokens < max_tokens) / len(num_tokens)


# 用来将tokens转换为文本
def reverse_tokens(tokens):
    text = ''
    for i in tokens:
        if i != 0:
            text = text + cn_model.index2word[i]
        else:
            text = text + ' '
    return text


reverse = reverse_tokens(train_tokens[1])
# 由此可见每一个词都对应一个长度为300的向量
embedding_dim = 300


# 只使用前20000个词
num_words = 15538
# 初始化embedding_matrix，之后在keras上进行应用
embedding_matrix = np.zeros((num_words, embedding_dim))
# embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
# 维度为 50000 * 300
for i in range(num_words):
    embedding_matrix[i, :] = cn_model[cn_model.index2word[i]]
embedding_matrix = embedding_matrix.astype('float32')
print(embedding_matrix)

# 检查index是否对应
# 输出300意义为长度为300的embedding向量一一对应
np.sum(cn_model[cn_model.index2word[333]] == embedding_matrix[333])


# embedding_matrix的维度，
# 这个维度为keras的要求，后续会在模型中用到
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


# 进行padding和truncating， 输入的train_tokens是一个list
# 返回的train_pad是一个numpy array
train_pad = pad_sequences(train_tokens, maxlen=max_tokens,
                          padding='pre', truncating='pre')

# 超出五万个词向量的词用0代替# 超出五万个词
train_pad[train_pad >= num_words] = 0
# 可见padding之后前面的tokens全变成0，文本在最后面# 可见padd
# 准备target向量，前7769样本为1，后7769为0
train_target = np.concatenate((np.ones(len(pos_txts)), np.zeros(len(pos_txts))))
# 80%的样本用来训练，剩余20%用来测试
X_train, X_test, y_train, y_test = train_test_split(train_pad,
                                                    train_target,
                                                    test_size=0.2,
                                                    random_state=12)
print(len(X_test))

# 查看训练样本，确认无误# 查看训练样本
print(reverse_tokens(X_test[0]))
print('class: ', y_test[0])


# 基于CNN进行情感分析
# 清空图
tf.reset_default_graph()
# 我在这里定义了3种filter，每种100个
filters_size = [2, 3]
num_filters = 100
# 超参数
BATCH_SIZE = 128
EPOCHES = 50
LEARNING_RATE = 0.001
L2_LAMBDA = 10
KEEP_PROB = 0.8

# X_train, X_test, y_train, y_test
def get_batch(x, y, batch_size=BATCH_SIZE, shuffle=True):
    assert x.shape[0] == y.shape[0], print("error shape!")
    # shuffle
    if shuffle:
        shuffled_index = np.random.permutation(range(x.shape[0]))

        x = x[shuffled_index]
        y = y[shuffled_index]

    # 统计共几个完整的batch
    n_batches = int(x.shape[0] / batch_size)

    for i in range(n_batches - 1):
        x_batch = x[i * batch_size: (i + 1) * batch_size]
        y_batch = y[i * batch_size: (i + 1) * batch_size]

        yield x_batch, y_batch


static_embeddings = embedding_matrix
print(static_embeddings)
EMBEDDING_SIZE = embedding_dim
# 句子最大长度
SENTENCE_LIMIT_SIZE = max_tokens


with tf.name_scope("cnn"):
    with tf.name_scope("placeholders"):
        inputs = tf.placeholder(dtype=tf.int32, shape=(None, max_tokens), name="inputs")
        targets = tf.placeholder(dtype=tf.int64, shape=[None], name="targets")
        y_one_hot = tf.one_hot(targets, 1)  # 正负分类
    # embeddings
    with tf.name_scope("embeddings"):
        embedding_matrixs = tf.Variable(initial_value=static_embeddings, trainable=False, name="embedding_matrixs")
        embed = tf.nn.embedding_lookup(embedding_matrix + 1, inputs, name="embed")
        # 添加channel维度
        embed_expanded = tf.expand_dims(embed, -1, name="embed_expand")
    # 用来存储每种filter的卷积池化max-pooling的结果
    pooled_outputs = []

    # 迭代多个filter
    for i, filter_size in enumerate(filters_size):
        with tf.name_scope("conv_maxpool_%s" % filter_size):
            filter_shape = [filter_size, EMBEDDING_SIZE, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, mean=0.0, stddev=0.1), name="W")
            b = tf.Variable(tf.zeros(num_filters), name="b")
            conv = tf.nn.conv2d(input=embed_expanded,
                                filter=W,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="conv")

            # 激活
            a = tf.nn.relu(tf.nn.bias_add(conv, b), name="activations")
            # 池化
            max_pooling = tf.nn.max_pool(value=a,
                                         ksize=[1, SENTENCE_LIMIT_SIZE - filter_size + 1, 1, 1],
                                         strides=[1, 1, 1, 1],
                                         padding="VALID",
                                         name="max_pooling")
            pooled_outputs.append(max_pooling)

    # 统计所有的filter，用于连接全连接层
    total_filters = num_filters * len(filters_size)
    total_pool = tf.concat(pooled_outputs, 3)
    flattend_pool = tf.reshape(total_pool, (-1, total_filters))

    # dropout
    with tf.name_scope("dropout"):
        dropout = tf.nn.dropout(flattend_pool, KEEP_PROB)

    # output
    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=(total_filters, 1), initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros(1), name="b")
        logits = tf.add(tf.matmul(dropout, W), b)
        predictions = tf.nn.sigmoid(logits, name="predictions")

    # loss，loss上添加了全连接层权重W的L2正则
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_one_hot, logits=logits))
        loss = loss + L2_LAMBDA * tf.nn.l2_loss(W)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # evaluation
    with tf.name_scope("evaluation"):
        # tf.greater 功能：通过比较a、b两个值的大小来输出对错。大于0.5即输出为负向文本
        correct_preds = tf.equal(tf.cast(tf.greater(predictions, 0.5), tf.float32), y_one_hot)
        accuracy = tf.reduce_sum(tf.reduce_sum(tf.cast(correct_preds, tf.float32), axis=1))


# 存储准确率
cnn_train_accuracy = []
cnn_test_accuracy = []
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./graphs/cnn", tf.get_default_graph())
    n_batches = int(X_train.shape[0] / BATCH_SIZE)
    for epoch in range(EPOCHES):
        total_loss = 0
        for x_batch, y_batch in get_batch(X_train, y_train):
            _, l = sess.run([optimizer, loss],
                            feed_dict={inputs: x_batch,
                                       targets: y_batch})
            total_loss += l

        train_corrects = sess.run(accuracy, feed_dict={inputs: X_train, targets: y_train})
        train_acc = train_corrects / X_train.shape[0]
        cnn_train_accuracy.append(train_acc)

        test_corrects = sess.run(accuracy, feed_dict={inputs: X_test, targets: y_test})
        test_acc = test_corrects / X_test.shape[0]
        cnn_test_accuracy.append(test_acc)

        print(
            "Training epoch: {}, Training loss: {:.4f}, Train accuracy: {:.4f}, Test accuracy: {:.4f}".format(epoch + 1,
                                                                                                              total_loss / n_batches,
                                                                                                              train_acc,
                                                                                                              test_acc))

    saver.save(sess, "checkpoints/cnn")
    # 保存二进制模型
    const_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['cnn/output/predictions'])
    # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['cnn/output/predictions'])
    with tf.gfile.FastGFile(r'Text_Sentiment.pb', mode='wb') as f:
        f.write(const_graph.SerializeToString())
    writer.close()


plt.plot(cnn_train_accuracy)
plt.plot(cnn_test_accuracy)
plt.ylim(ymin=0.5, ymax=1.01)
plt.title("The accuracy of CNN model")
plt.legend(["train", "test"])


# 在test上的准确率
with tf.Session() as sess:
    saver.restore(sess, "checkpoints/cnn")
    feed = {inputs: X_test, targets: y_test}
    total_correct = sess.run(accuracy,
                             feed_dict=feed)
    print("判断正确的文本数量：" + str(total_correct))
    print("测试文本总量：" + str(X_test.shape[0]))
    print("The textCNN model accuracy on test set: {:.2f}%".format(100 * total_correct / X_test.shape[0]))
    preds = sess.run(predictions, feed_dict={inputs: X_test})
    print(len(X_test[0]))


def predict_sentiment(text):
    # print(text)
    # 去标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
    # 分词
    cut = jieba.cut(text)
    cut_list = [i for i in cut]
    # tokenize
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            cut_list[i] = 0
    # padding
    tokens_pad = pad_sequences([cut_list], maxlen=max_tokens,
                               padding='pre', truncating='pre')
    # 超出五万个词向量的词用0代替# 超出五万个词
    tokens_pad[tokens_pad >= num_words] = 0
    # print(tokens_pad)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('checkpoints/cnn.meta')
        saver.restore(sess, tf.train.latest_checkpoint("checkpoints/"))
        result = sess.run(predictions, feed_dict={inputs: tokens_pad})
        # 预测
        # result = model.predict(x=tokens_pad)
        coef = result[0][0]
        print(coef)
        if coef >= 0.5:
            print('===========================是一例负面新闻==================', 'output=%.2f' % coef)
        else:
            print('===========================是一例正面新闻==================', 'output=%.2f' % coef)








