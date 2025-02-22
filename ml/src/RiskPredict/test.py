import pandas as pd

load_data = pd.read_excel("./dataSets/DEA三级到二级.xlsx", encoding="gbk")
raw_data = load_data.fillna(0)
col_cz = ["营业收入增长率", "总资产增长率", "净资产收益率", "总资本盈利率", "现金收入比"]
# col_czi = ["纳税信用等级","财务费用率","产权比率","速动比率","现金比率","现金流量债务比"]
# col_ld = ["流动资产周转率","应收账款周转率","流动资产合计","非流动资产合计","流动负债合计","营运资本周转率"]
# col_qc = ["流动资产", "流动负债", "资产总计", "净利润", "所得税", "经营现金流量净额"]
# 各指标特征
CZ_features = raw_data.loc[:, col_cz]
# CZI_features =  raw_data.loc[:, col_czi]
# LD_features  =  raw_data.loc[:, col_ld]
# QC_features  =  raw_data.loc[:, col_qc]
# 各指标 输出
CZ_y = raw_data.loc[:, ["成长力"]]
# CZI_y = raw_data.loc[:, ["筹资力"]]
# LD_y = raw_data.loc[:, ["流动性"]]
# QC_y = raw_data.loc[:, ["清偿力"]]


# 切分训练测试数据
from sklearn.model_selection import train_test_split


def split_x_y(x, y_data):
    x_train, x_test, y_train, y_test = train_test_split(
    x, y_data, train_size=0.8, random_state=1)  # 参数test_size设置训练集占比
    return x_train, x_test, y_train, y_test


# 成长力数据切分
from sklearn import preprocessing


inputs_x = CZ_features.values
output_y = CZ_y.values
x_train, x_test, y_train, y_test = split_x_y(inputs_x, output_y)
#X_scaled = preprocessing.scale(y_train)
#  该类的好处在于可以保存训练集中的参数（均值、方差）直接使用其对象转换测试集数据
ss_x = preprocessing.StandardScaler(y_train)
# 先拟合再标准化训练集数据
# train_x_disorder = ss_x.fit_transform(inputs_x)
# # 使用上面所得均值和方差直接归一化测试集数据
# test_x_disorder = ss_x.transform(inputs_x)
# ss_y = preprocessing.StandardScaler()
# train_y_disorder = ss_y.fit_transform(y_train.reshape(-1, 1))
print(ss_x.mean(axis=0))
print(ss_x.std(axis=0))
print(-0.014020368)
# test_y_disorder = ss_y.transform(y_test.reshape(-1, 1))

