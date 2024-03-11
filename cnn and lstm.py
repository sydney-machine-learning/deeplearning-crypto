# -*- coding: utf-8 -*-
"""
@file: 
@author: 
@time:
@description:


"""

# 读取数据
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

def read_data(dim_type):
    df = pd.read_csv(r"coin_Bitcoin.csv")
    data_len = df.shape[0]
    if dim_type == 'Multi':
        open_data = df["Open"].values.reshape((data_len, 1))
        high_data = df["High"].values.reshape((data_len, 1))
        low_data = df["Low"].values.reshape((data_len, 1))
        close_data = df["Close"].values.reshape((data_len, 1))
        marketcap_data = df["Marketcap"].values.reshape((data_len, 1))
        dataset = np.hstack((close_data, open_data, high_data, low_data, marketcap_data)) # 选取多个维度，并将close_data置于第一列
        data_len = df.shape[0]
        return dataset, data_len
    else:
        dim_data = df[dim_type].values.reshape((data_len, 1))
        data_len = df.shape[0]
        return dim_data, data_len

def n_fea(dim_type):
    if dim_type == 'Multi':
        return 4
    else:
        return 1

def split_sequence(sequence, dim_type, n_steps_in, n_steps_out):
    X, y = list(), list()
    if dim_type == 'Multi':
        for i in range(len(sequence)):
            # find the end of the input pattern
            end_ix = i + n_steps_in
            # find the end of the output pattern
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(sequence):
                break
            # gather input and output parts of the pattern
            seq_x = sequence[i:end_ix, 1:]
            seq_y = sequence[end_ix:out_end_ix, 0]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    else:
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the sequence
            if out_end_ix > len(sequence):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

def MinMaxdata(data):
# 创建一个空字典来存储每一列的 scaler
    scalers = {}
    # 归一化数据的容器
    normalized_data = np.zeros_like(data)
    # 循环每一列
    for i in range(data.shape[1]):  # data.shape[1] 是列的数量
        # 为每一列创建一个新的 MinMaxScaler
        scaler = MinMaxScaler()
        # 将列数据调整为正确的形状，即(-1, 1)
        column_data = data[:, i].reshape(-1, 1)
        # 拟合并转换数据
        normalized_column = scaler.fit_transform(column_data)
        # 将归一化的数据存回容器中
        normalized_data[:, i] = normalized_column.ravel()
        # 存储scaler以便后续使用
        scalers[i] = scaler
    # 现在 normalized_data 是完全归一化的数据
    # scalers 字典包含每一列的 MinMaxScaler 实例
    return normalized_data, scalers

def anti_MinMaxdata(data, scaler):
    # 如果data是三维数组，去除最后一个维度
    if data.ndim == 3 and data.shape[2] == 1:
        data = data.squeeze(axis=2)

    restored_data = np.zeros_like(data)
    for i in range(data.shape[1]):  # 遍历所有列
        column_data = data[:, i].reshape(-1, 1)
        restored_data[:, i] = scaler.inverse_transform(column_data).ravel()
    return restored_data

def create_lstm_model(dim_type, n_steps_in, n_steps_out):
    n_features = n_fea(dim_type)
    model = Sequential()
    model.add(LSTM(100, activation='sigmoid', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(100, activation='sigmoid'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    return model

def create_bidirectional_lstm_model(dim_type, n_steps_in, n_steps_out):
    n_features = n_fea(dim_type)
    model = Sequential()
    model.add(Bidirectional(LSTM(100, activation='sigmoid'), input_shape=(n_steps_in, n_features)))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    return model

def create_ED_lstm_model(dim_type, n_steps_in, n_steps_out):
    n_features = n_fea(dim_type)
    model = Sequential()
    model.add(LSTM(100, activation='sigmoid', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(100, activation='sigmoid'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    return model

def lstm_train_model_and_forecast(model, dim_type, data_X, data_Y, n_steps_in, n_steps_out, ech):
    n_features = n_fea(dim_type)
    X, y = split_sequence(data_X, dim_type, n_steps_in, n_steps_out)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # 训练模型
    model.fit(X, y, epochs=ech, verbose=1)

    # 拟合模型
    fit_result = []
    for index, ele in enumerate(X):
        pred = model.predict(ele.reshape((1, n_steps_in, n_features)))
        fit_result.append(pred)
    fr = np.array(fit_result)
    fit_result = fr.reshape(len(fit_result), n_steps_out)
    
    # 测试模型
    test_x, test_y = split_sequence(data_Y, dim_type, n_steps_in, n_steps_out)
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], n_features))
    test_result = []
    for index, ele in enumerate(test_x):
        pred = model.predict(ele.reshape((1, n_steps_in, n_features)))
        test_result.append(pred)
    tr = np.array(test_result)
    test_result = tr.reshape(len(test_result), n_steps_out)
    return fit_result, test_result

def create_CNN_model(dim_type, n_steps_in, n_steps_out):
    n_features = n_fea(dim_type)
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def cnn_train_model_and_forecast(model, dim_type, data_X, data_Y, n_steps_in, n_steps_out, ech):
    n_features = n_fea(dim_type)
    X, y = split_sequence(data_X, dim_type, n_steps_in, n_steps_out)

    # 对于多维数据，调整最后一个维度为特征数
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    # 训练模型
    model.fit(X, y, epochs=ech, batch_size=32, verbose=1)

    # 拟合模型
    fit_result = []
    for index, ele in enumerate(X):
        pred = model.predict(ele.reshape((1, n_steps_in, n_features)))
        fit_result.append(pred)
    fr = np.array(fit_result)
    fit_result = fr.reshape(len(fit_result), n_steps_out)
    
    # 测试模型
    test_x, test_y = split_sequence(data_Y, dim_type, n_steps_in, n_steps_out)
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], n_features))
    test_result = []
    for index, ele in enumerate(test_x):
        pred = model.predict(ele.reshape((1, n_steps_in, n_features)))
        test_result.append(pred)
    tr = np.array(test_result)
    test_result = tr.reshape(len(test_result), n_steps_out)
    return fit_result, test_result

def calculate_rmse(result, n_steps_out, target):
    # 检查输入数组长度是否相等
    # 归一化
    result, _ = MinMaxdata(result)
    target, _ = MinMaxdata(target)
    rmse = []
    for i in range(n_steps_out):
        rmse.append(np.sqrt(np.mean((result[:, i] - target[:, i]) ** 2)))
    return rmse

def calculate_mape(result, n_steps_out, target):
    # 检查输入数组长度是否相等
    mape = []
    for i in range(n_steps_out):
        mape.append(np.mean(np.abs((target[:, i] - result[:, i]) / target[:, i])) * 100)
    return mape

def main():
    # 准备数据
    data, data_len = read_data(dim_type) # 读取数据
    data, scalers = MinMaxdata(data) # 归一化数据
    data_X = data[0:int(np.floor(data_len * percentage))] # 训练集
    data_Y = data[int(np.floor(data_len * percentage)):] # 测试集

    # LSTM模型
    lstm_model = create_lstm_model(dim_type, n_steps_in, n_steps_out)
    fit_result_lstm, test_result_lstm = lstm_train_model_and_forecast(
        lstm_model, dim_type, data_X, data_Y, n_steps_in, n_steps_out, epochs)
    fit_result_lstm = anti_MinMaxdata(fit_result_lstm, scalers[0]) # 反归一化
    test_result_lstm = anti_MinMaxdata(test_result_lstm, scalers[0]) # 反归一化

    # Bidirectional LSTM模型
    bidirectional_lstm_model = create_bidirectional_lstm_model(dim_type, n_steps_in, n_steps_out)
    fit_result_bid_lstm, test_result_bid_lstm = lstm_train_model_and_forecast(
        bidirectional_lstm_model, dim_type, data_X, data_Y, n_steps_in, n_steps_out, epochs)
    fit_result_bid_lstm = anti_MinMaxdata(fit_result_bid_lstm, scalers[0]) # 反归一化
    test_result_bid_lstm = anti_MinMaxdata(test_result_bid_lstm, scalers[0]) # 反归一化

    # ED LSTM模型
    ED_lstm_model = create_ED_lstm_model(dim_type, n_steps_in, n_steps_out)
    fit_result_ED_lstm, test_result_ED_lstm = lstm_train_model_and_forecast(
        ED_lstm_model, dim_type, data_X, data_Y, n_steps_in, n_steps_out, epochs)
    fit_result_ED_lstm = anti_MinMaxdata(fit_result_ED_lstm, scalers[0]) # 反归一化
    test_result_ED_lstm = anti_MinMaxdata(test_result_ED_lstm, scalers[0]) # 反归一化

    # CNN模型
    CNN_model = create_CNN_model(dim_type, n_steps_in, n_steps_out)
    fit_result_CNN, test_result_CNN = cnn_train_model_and_forecast(
        CNN_model, dim_type, data_X, data_Y, n_steps_in, n_steps_out, epochs)
    fit_result_CNN = anti_MinMaxdata(fit_result_CNN, scalers[0]) # 反归一化
    test_result_CNN = anti_MinMaxdata(test_result_CNN, scalers[0]) # 反归一化

    # 计算RMSE
    _ , fit = split_sequence(data_X, dim_type, n_steps_in, n_steps_out)
    fit = anti_MinMaxdata(fit, scalers[0]) # 反归一化
    _ , test = split_sequence(data_Y, dim_type, n_steps_in, n_steps_out)
    test = anti_MinMaxdata(test, scalers[0]) # 反归一化

    fit_rmse_lstm = calculate_rmse(fit_result_lstm, n_steps_out, fit)
    test_rmse_lstm = calculate_rmse(test_result_lstm, n_steps_out, test)
    print('LSTM Fit Rmse:', fit_rmse_lstm)
    print('LSTM Test Rmse:', test_rmse_lstm)

    fit_rmse_bid_lstm = calculate_rmse(fit_result_bid_lstm, n_steps_out, fit)
    test_rmse_bid_lstm = calculate_rmse(test_result_bid_lstm, n_steps_out, test)
    print('Bidirectional LSTM Fit Rmse:', fit_rmse_bid_lstm)
    print('Bidirectional LSTM Test Rmse:', test_rmse_bid_lstm)

    fit_rmse_ED_lstm = calculate_rmse(fit_result_ED_lstm, n_steps_out, fit)
    test_rmse_ED_lstm = calculate_rmse(test_result_ED_lstm, n_steps_out, test)
    print('ED LSTM Fit Rmse:', fit_rmse_ED_lstm)
    print('ED LSTM Test Rmse:', test_rmse_ED_lstm)

    fit_rmse_CNN = calculate_rmse(fit_result_CNN, n_steps_out, fit)
    test_rmse_CNN = calculate_rmse(test_result_CNN, n_steps_out, test)
    print('CNN Fit Rmse:', fit_rmse_CNN)
    print('CNN Test Rmse:', test_rmse_CNN)

    # 绘图
    if n_steps_out == 1:
        # 拟合结果
        plt.figure(figsize=(15, 6))
        plt.plot(fit_result_lstm, label='LSTM')
        plt.plot(fit_result_bid_lstm, label='Bidirectional LSTM')
        plt.plot(fit_result_ED_lstm, label='ED LSTM')
        plt.plot(fit_result_CNN, label='CNN')
        plt.plot(fit, label='Actual')
        plt.title('Fit Results')
        plt.xlabel('Time Steps')
        plt.ylabel('Open Price')
        plt.legend()
        plt.show()
        
        # 测试结果
        plt.figure(figsize=(15, 6))
        plt.plot(test_result_lstm, label='LSTM')
        plt.plot(test_result_bid_lstm, label='Bidirectional LSTM')
        plt.plot(test_result_ED_lstm, label='ED LSTM')
        plt.plot(test_result_CNN, label='CNN')
        plt.plot(test, label='Actual')
        plt.title('Test Results')
        plt.xlabel('Time Steps')
        plt.ylabel('Open Price')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    # parameters
    dim_type = 'Close' # 'Multi' or 'Open', 'High', 'Low', 'Close', 'Marketcap' (选取数据的维度或类型)

    n_steps_in = 5 # 输入步长
    n_steps_out = 2 # 输出步长
    percentage = 0.7 # 训练集百分比
    epochs = 200 # 迭代次数
    
    main() 