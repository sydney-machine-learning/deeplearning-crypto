import numpy as np
import keras

# 读取数据
import pandas as pd
import keras
import numpy as np
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# 这个是分割二维数据的
""" def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix], sequences[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y) """


n_steps = 3
n_features = 1
n_steps_in = 3
n_steps_out = 1
# choose a number of time steps


# 使用 pandas 读取CSV文件
df = pd.read_csv(r"coin_Bitcoin.csv")
data_len = df.shape[0]

# 提取 "Open" 列的数据
open_data = df["Open"].values.reshape((data_len, 1))
high_data = df["High"].values.reshape((data_len, 1))
low_data = df["Low"].values.reshape((data_len, 1))
close_data = df["Close"].values.reshape((data_len, 1))
marketcap_data = df["Marketcap"].values.reshape((data_len, 1))

# open_data 为预测y值(改完之后这段没啥用)
dataset = hstack((high_data, low_data, close_data, marketcap_data))
# 初始化归一化器
scalerdata = MinMaxScaler(feature_range=(0, 1))
# 归一化 open_data
data_normalized = scalerdata.fit_transform(dataset)

# 初始化归一化器
scaler = MinMaxScaler(feature_range=(0, 1))
# 归一化 open_data
open_data_normalized = scaler.fit_transform(open_data)

data_normalized = hstack((data_normalized, open_data_normalized))

# 切分数据集7：3
train_data = data_normalized[0:int(data_len * 0.7)]
test_data = data_normalized[int(data_len * 0.7):]

# 训练
train_x, train_y = split_sequences(train_data, n_steps)
n_features = train_x.shape[2]
# 定义模型
model = Sequential()
model.add(LSTM(100, activation='sigmoid', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(train_x, train_y, epochs=100, verbose=0)

# 测试
test_x, test_y = split_sequences(test_data, n_steps)
total = len(test_y)
result = []
for index, ele in enumerate(test_x):
    # print(f"index:{index} /{total}")
    pred = model.predict(ele.reshape((1, n_steps, n_features)))
    result.append(pred)

# 评估
result_y = np.array(result)
# 计算rmse
rmse = np.sqrt(np.mean((result_y - test_y) ** 2))
print(f"rmse:{rmse}")


result_y_flat = np.array(result_y).squeeze()

# 反归一化预测结果
result_y_flat_original = scaler.inverse_transform(result_y_flat.reshape(-1, 1))
# 反归一化测试集的目标值
test_y_original = scaler.inverse_transform(test_y.reshape(-1, 1))

# 绘制反归一化后的结果
plt.figure(figsize=(12, 6))
plt.plot(result_y_flat_original, label='Predicted', color='blue')
plt.plot(test_y_original, label='Actual', color='orange')
plt.title('Comparison of Predicted and Actual Open Prices (Original Scale)')
plt.xlabel('Time Steps')
plt.ylabel('Open Price')
plt.legend()
plt.show()
