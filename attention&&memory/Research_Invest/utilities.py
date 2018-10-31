import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


def standardize(train, test):
    """ Standardize data """
    # Standardize train and test
    X_train = (train - np.mean(train, axis=0)[None, :, :]) / np.std(train, axis=0)[None, :, :]
    X_test = (test - np.mean(test, axis=0)[None, :, :]) / np.std(test, axis=0)[None, :, :]

    return X_train, X_test


def one_hot(labels, n_class):
    """ One-hot encoding """
    expansion = np.eye(n_class)
    y = expansion[:, labels].T
    #assert y.shape[1] == n_class, "Wrong number of labels!"    
    return y


def get_batches(X, y, batch_size=50):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches * batch_size], y[:n_batches * batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b + batch_size], y[b:b + batch_size]

#def get_train_test_data( batch_size, file_path, time_step = 60):
    """ Return train and test data """
    
    