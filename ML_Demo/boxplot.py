import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scio

input = "D://java-project/enterpriseInfo/datasets/Data554.mat";
data = scio.loadmat(input);
matrix = data['Data']
print(matrix)