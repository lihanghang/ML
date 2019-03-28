# _*_ coding:utf-8 -*_
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit

def read_data(url):
    data = pd.read_csv(url)
    dummies = pd.get_dummies(data['group1'],   drop_first=False)
   #print(dummies)
    
    label = data['label']
    weight = data['weight']
    X = data.drop(['group1','era','id','weight','label'],axis=1)
    return X, dummies, weight, label

def scale_feature(X,dummies,quantile_percent=0.9):
    scaled_features = {}
    for each in X.columns:
        mean, std = X[each].mean(), X[each].std()
        scaled_features[each] = [mean,std]
        X.loc[:, each] = (X[each] - mean)/std
        X.loc[X[each]>X[each].quantile(quantile_percent)] = X[each].quantile(quantile_percent)
    X = pd.concat([X, dummies], axis=1)
    return X, scaled_features

def data_split(X,Y,test_size=0.2):
    values = X.values
    labels = Y.values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    for train_index, test_index in sss.split(values,labels):
        X_train, Y_train = values[train_index], labels[train_index]
        X_test, Y_test = values[test_index], labels[test_index]
    return X_train, Y_train, X_test, Y_test
X, dummies, weight, label = read_data(r'./stock_train_data_20171125.csv')
X, scaled_features = scale_feature(X,dummies,quantile_percent=0.995)

X_train, Y_train, X_test, Y_test = data_split(X, label, test_size=0.1)
print('X_train shape:',X_train.shape,'\n',
     'Y_train shape:', Y_train.shape,'\n',
     'X_test shape:', X_test.shape,'\n',
     'Y_test shape:', Y_test.shape)
# data = pd.read_csv(r'C:\Users\Administrator\Anaconda3\stock_train_data_20171125.csv')
# dummies = pd.get_dummies(data['groups1'], prefix='groups')
# label = data['label']
# weight = data['weight']
# X = data.drop(['groups1','era','id','weight','label'],axis=1)  
# print(info)
def get_batches(X, Y, batch_size):
    data_len = len(X)
    for i in range(0, data_len, batch_size):
        end = i + batch_size
        if end > data_len:
            end = -1
        x = X[i: end].reshape(-1,X.shape[1])
        #print(x.shape)
        y = Y[i : end].reshape(-1,1)
        yield x, y

def build_inputs(num_features):
    '''
    构建输入
    '''
    inputs = tf.placeholder(tf.float32, [None, num_features], name='inputs')
    targets = tf.placeholder(tf.float32, [None, 1], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, keep_prob

def fc_model(inputs,keep_prob):
    layer1 = tf.layers.dense(inputs,58,activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer())
    dropout = tf.nn.dropout(layer1,keep_prob)
    layer2 = tf.layers.dense(dropout,29,activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer())
    dropout = tf.nn.dropout(layer2,keep_prob)
    layer3 = tf.layers.dense(dropout,14,activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer())
    dropout = tf.nn.dropout(layer3,keep_prob)
    logits = tf.layers.dense(dropout,1,activation=None,kernel_initializer=tf.truncated_normal_initializer(), name='logits')
    return logits


def train(X_train,Y_train,X_test,Y_test,keep_prob,epoch_count, batch_size, learning_rate=0.001, num_features=108):
    inputs, targets, k_p = build_inputs(num_features)
    logits = fc_model(inputs,k_p)
    out = tf.sigmoid(logits)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=targets))
    train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_pred = tf.equal(tf.cast(tf.round(out), tf.int32), tf.cast(targets, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    steps = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for x,y in get_batches(X_train,Y_train,batch_size):
                steps += 1
                _, train_loss, train_accuracy = sess.run([train_opt, loss, accuracy], feed_dict={inputs:x, targets:y, k_p:keep_prob})
                
                if steps % 1000 == 0:
                    test_loss, test_accuracy = sess.run([loss, accuracy], feed_dict={inputs:X_test.reshape(-1,num_features),
                                                                                     targets:Y_test.reshape(-1,1), k_p:1.0})
                    print("Epoch {}/{}.".format(epoch_i+1, epoch_count),
                          "train_loss: {:.4f}..".format(train_loss),
                          "train_acc: {:.4f}..".format(train_accuracy),
                          "test_loss:{:.4f}..".format(test_loss),
                          "test_acc:{:.4f}..".format(test_accuracy))
                    
        data = pd.read_csv(r'C:\Users\Administrator\Anaconda3\stock_test_data_20171125.csv')
        #print("info:",data)
        dummies = pd.get_dummies(data['group1'], prefix='group', drop_first=False)
        X = data.drop(['group1','id'],axis=1)
        for each in X.columns:
            X.loc[:, each] = (X[each] - scaled_features[each][0])/scaled_features[each][1]
            X.loc[X[each]>X[each].quantile(0.995)] = X[each].quantile(0.995)
                              
        X = pd.concat([X, dummies], axis=1).values
        output = sess.run(out, feed_dict={inputs:X.reshape(-1,128),k_p:1.0})
        print(len(output))
        print(len(data))
        data['proba'] = output
        data[['id','proba']].to_csv('proba.csv',index=False)

batch_size = 1000
learning_rate = 0.0003
keep_prob = 0.80
epochs = 800

with tf.Graph().as_default():
    train(X_train,Y_train,X_test,Y_test,keep_prob,epochs,batch_size,learning_rate)
