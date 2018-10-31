# coding: utf-8

# import pandas as pd
# import numpy as np
# from numpy.random import RandomState


# # 经营、司法(0.11595)

# raw_data = pd.read_csv("./dataSets/raw_data.csv", encoding="gbk")
# out_data = pd.read_csv("./dataSets/output_data.csv", encoding="gbk")
# # # 竞争风险输出值
# Y = out_data.loc[:,["司法"]]
# y_data = np.array(Y)#np.ndarray()
# # y_list =y_data.tolist()#list
# # Y.head()


# In[ ]:


# 绘制误差曲线
# t = np.arange(iteration-1)
# plt.figure(figsize = (9,6))
# plt.plot(t, np.array(error), 'b*')
# plt.xlabel("iteration")
# plt.ylabel("error")
# plt.legend(['error'], loc='upper right')
# plt.show()

# 使用线性回归的方法进行数据曲线拟合
# from sklearn import linear_model
# from sklearn.externals import joblib

# mlr = linear_model.LinearRegression()

# model = mlr.fit(x_train, y_train)
# joblib.dump(model,'rf.model')

# print(mlr)
# print ("coef:")
# print(mlr.coef_)
# print ("intercept")
# print( mlr.intercept_)

# #xPredict =  x_test[0].tolist()
# # # xPredict
# yPredict = mlr.predict(x_test)
# # # # print "predict:"
# print(yPredict)


# In[ ]:


# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
# from tensorflow.python.framework import graph_util



# # 竞争风险输入指标2;经营力：6
# #col_n = ['对外投资数量','竞品信息']
# #col_n = ['经营力方面---财务风险-盈利能力-销售净利率','财务风险-盈利能力-总资产报酬率', '财务风险-盈利能力-净资产收益率','财务风险-资产利用-存货周转率'
# #        ,'财务风险-资产利用-总资产周转率', '财务风险-资产利用-成本费用利润率']

# col_n = ["司法方面---经营风险-司法风险-诉讼数量","经营风险-其它风险-经营异常次数","经营风险-其它风险-行政处罚次数","经营风险-其它风险-动产质押次数","经营风险-其它风险-自身风险","经营风险-其它风险-周边风险"]
# competition = raw_data.loc[:, col_n].values
# #x_3 =  competition[:,3:6]

# #x=np.column_stack([competition,x_3])#随意给x增加了3列，x变为16列，可以reshape为3*3矩阵了 没啥用，就是凑个正方形


# # .replace([0, 0],[4,5]
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y_data, train_size=0.8, random_state=123) # 参数test_size设置训练集占比


# In[1]:


import tensorflow as tf
import os

# 使用GPU训练
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("tensorflow运行版本：" + tf.__version__)


# CNN 回归预测
sess = tf.Session()


#  权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="w")


#  偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="b")


# 卷积层
def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


#  池化层
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# define placeholder for inputs to network
xs = tf.placeholder("float", shape=[None, 6], name='input_x') #原始数据的维度：6
ys = tf.placeholder("float", shape=[None, 1])  # 输出数据为维度：1

keep_prob = tf.placeholder(tf.float32)  # dropout的比例

# -1 数据数量不定2*3*1通道为1
x_image = tf.reshape(xs, [-1, 2, 3, 1])  # 原始数据16变成二维图片4*4



#  conv1 layer  第一卷积层
W_conv1 = weight_variable([2, 2, 1, 12])  # 卷积核大小patch 2x2, 图像通道数in size 1, 卷积核数目out size 6,
#  每个卷积核对应一个偏置量
b_conv1 = bias_variable([12])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 2x2x32，长宽不变，高度为32的三维图像
#  h_pool1 = max_pool_2x2(h_conv1)     # output size 2x2x32 长宽缩小一倍


## conv2 layer  第二卷积层 6个通道卷积, 卷积出12个特征
W_conv2 = weight_variable([2, 2, 12, 24])  # patch 2x2, in size 32, out size 64
b_conv2 = bias_variable([24])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2) # 输入第一层的处理结果 输出shape 4*4*64

# fc1 layer ##  full connection 全连接层
W_fc1 = weight_variable([2*3*24, 144])  # 3x3 ，高度为64的三维图片，然后把它拉成512长的一维数组
b_fc1 = bias_variable([144])

h_pool2_flat = tf.reshape(h_conv2, [-1, 2*3*24])  # 把4*4，高度为64的三维图片拉成一维数组 降维处理
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)    # dropout层
# fc2 layer ## full connection

with tf.name_scope('output'):
    W_fc2 = weight_variable([144, 1])
    b_fc2 = bias_variable([1])  # 偏置
    # 最后的计算结果
    multi = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    pred = tf.add(multi, b_fc2, name="predict")

# 计算 predition与y 差距 所用方法很简单就是用 suare()平方,sum()求和,mean()平均值
cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys - pred), reduction_indices=[1]))
# 0.01学习效率,minimize(loss)减小loss误差
train_step = tf.train.AdamOptimizer(0.0035).minimize(cross_entropy)
sess.run(tf.global_variables_initializer())


# 计算平均误差
def Get_Average(list):
    sum = 0
    for item in list:
        sum += item
    return sum/len(list)


# 训练模型
def train_model(x_train, y_train, iteration, model_name):
    error = []
    for i in range(iteration):
        sess.run(train_step, feed_dict={xs: x_train, ys: y_train, keep_prob: 1.0})
        err = sess.run(cross_entropy, feed_dict={xs: x_train, ys: y_train, keep_prob: 1.0})
        print(i, '误差=', err)  # 输出loss值
        error.append(err)
    # 计算平均误差
    ave_error = Get_Average(error)
    print("平均误差为：%f " % ave_error)
    # 保存模型
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output/predict'])
    with tf.gfile.FastGFile(model_name, mode='wb') as f:
        f.write(output_graph_def.SerializeToString())
        print("Model Save succesfull!")


import pandas as pd


# 依次建立成长力、筹资力、流动性、清偿力四大财务风险指标；20180823午
# 加载全部数据包括四个模型数据,数据预处理
load_data = pd.read_excel("./dataSets/DEA三级到二级.xlsx", encoding="gbk")
raw_data = load_data.fillna(0)
col_cz = ["营业收入增长率","总资产增长率","净资产收益率","总资本盈利率","现金收入比","购建固定资产及无形资产和其他长期资产支付的现金（"]
col_czi = ["纳税信用等级","财务费用率","产权比率","速动比率","现金比率","现金流量债务比"]
col_ld = ["流动资产周转率","应收账款周转率","流动资产合计","非流动资产合计","流动负债合计","营运资本周转率"]
col_qc = ["流动资产", "流动负债", "资产总计", "净利润", "所得税", "经营现金流量净额"]
# 各指标特征
CZ_features  =  raw_data.loc[:, col_cz]
CZI_features =  raw_data.loc[:, col_czi]
LD_features  =  raw_data.loc[:, col_ld]
QC_features  =  raw_data.loc[:, col_qc]
# 各指标 输出
CZ_y = raw_data.loc[:, ["成长力"]]
CZI_y = raw_data.loc[:, ["筹资力"]]
LD_y = raw_data.loc[:, ["流动性"]]
QC_y = raw_data.loc[:, ["清偿力"]]


# 切分训练测试数据
from sklearn.model_selection import train_test_split


def split_x_y(x, y_data):
#     x_3 = x[:,3:6]
#     x_=np.column_stack([x,x_3])#随意给x增加了3列，x变为16列，可以reshape为3*3矩阵正方形
    x_train, x_test, y_train, y_test = train_test_split(
    x, y_data, train_size=0.8, random_state=1)  # 参数test_size设置训练集占比
    return x_train, x_test, y_train, y_test


# 成长力数据切分
from sklearn import preprocessing


inputs_x = LD_features.values
output_y = LD_y.values
x_train, x_test, y_train, y_test = split_x_y(inputs_x, output_y)

#  该类的好处在于可以保存训练集中的参数（均值、方差）直接使用其对象转换测试集数据
ss_x = preprocessing.StandardScaler()
# 先拟合再标准化训练集数据
train_x_disorder = ss_x.fit_transform(x_train)
# 使用上面所得均值和方差直接归一化测试集数据
test_x_disorder = ss_x.transform(x_test)
#  print(test_x_disorder)
ss_y = preprocessing.StandardScaler()
train_y_disorder = ss_y.fit_transform(y_train.reshape(-1, 1))
test_y_disorder = ss_y.transform(y_test.reshape(-1, 1))
#  print(test_y_disorder)

# 训练
train_model(train_x_disorder, train_y_disorder, 20000, "./Model/LD_Model.pb")


# 模型测试
import matplotlib
# 本地显示图像
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def model_test(x_test, y_test, index_name):
    prediction_value = sess.run(pred, feed_dict={xs: x_test, ys: y_test, keep_prob: 1.0})
    origin_data_y = y_test
    #print(origin_data_y)
    pred_data = ss_y.inverse_transform(prediction_value)
    # 绘图
    fig = plt.figure(figsize=(10, 10))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
    axes = fig.add_subplot(1, 1, 1)
    line1, = axes.plot(range(len(prediction_value)), pred_data, 'b--', label='pred', linewidth=2)
    line2, = axes.plot(range(len(origin_data_y)), origin_data_y, 'g', label='test')
    axes.grid()
    fig.tight_layout()
    plt.legend(handles=[line1, line2])
    plt.title(index_name)
    plt.show()
    # 打印预测值
    print(pred_data)
    #print(prediction_value)


# 预测
# test = [[ -2.21283262e-02,  -4.22827772e-02,  -1.22634684e-01,  -1.01158189e-01,
#    -1.16706481e-01,  -3.61754572e-02]]
# t = ss_x.fit_transform(test)
model_test(test_x_disorder, y_test, "CZ")
