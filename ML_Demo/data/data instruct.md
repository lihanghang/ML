## training process instruct
#### explain of train and test datas
1. id_num：代表时间每一个时间点，机器预测每一个时间点的PM2.5的值
2. test.csv为每个月后10天	的采集数据；train.csv为每个月前20天的数据，避免其他客观因素影响训练的效果
3. train.csv其中0--23表示第0时23时
4. output：prediction PM2.5 of xxxx year xx month xx day xx hour
#### training process
1. cross validation?
2. ……