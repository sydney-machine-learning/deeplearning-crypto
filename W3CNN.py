#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhc

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
# from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.layers import MaxPooling1D

# 读取数据
df = pd.read_csv('./DataUni/coin_Bitcoin.csv')
df = pd.DataFrame(df['Close'], columns=['Close'])
# Normalize the dataset
# 正则化数据
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(df)


# Function to create the dataset for the CNN
def create_dataset(data, time_step=1):
    X_data, y_data = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X_data.append(a)
        y_data.append(data[i + time_step, 0])
    return np.array(X_data), np.array(y_data)


# 时间步数
time_step = 3
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data into train and test sets
# 把数据集分为train和text 前70%是训练集
train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Build the CNN model
# 构建CNN模型
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(time_step, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

train_multi_predict = pd.DataFrame(index=range(len(X_train)))
test_multi_predict = pd.DataFrame(index=range(len(X_test)))
# i=1

for i in range(10):
    # Train the model
    # 训练模型
    model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=1)

    # Predictions
    # 预测
    train_multi_predict[i] = model.predict(X_train)
    test_multi_predict[i] = model.predict(X_test)

train_predict = np.reshape(train_multi_predict.mean(axis=1), (len(X_train), 1))
test_predict = np.reshape(test_multi_predict.mean(axis=1), (len(X_test), 1))
# Invert predictions back to original scale
train_predict_inv = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform([y_train])
test_predict_inv = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform([y_test])

# 计算训练集RMSE
train_rmse = np.sqrt(np.mean((y_train.flatten() - train_predict.flatten()) ** 2))
print("Train RMSE:", train_rmse)
# 计算测试集RMSE
test_rmse = np.sqrt(np.mean((y_test.flatten() - test_predict.flatten()) ** 2))
print("Test RMSE:", test_rmse)

# Plotting the results
# 画图 训练集的预测值和实际值
plt.figure(figsize=(15, 6))
plt.plot(y_train_inv.flatten(), label='Train actual')
plt.plot(train_predict_inv.flatten(), label='Train predictions')
plt.legend()
plt.savefig('./train_predictions.png', dpi=300)
plt.show()

# 测试集的预测值和实际值
plt.figure(figsize=(15, 6))
plt.plot(y_test_inv.flatten(), label='Test actual')
plt.plot(test_predict_inv.flatten(), label='Test predictions')
plt.legend()
plt.savefig('./test_predictions.png', dpi=300)
plt.show()
