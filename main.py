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
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Conv1D, Flatten, MaxPooling1D, RepeatVector, \
    TimeDistributed
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt



def read_data(dim_type,use_percentage):
    df = pd.read_csv(r"coin_Bitcoin.csv")
    data_len = df.shape[0]
    data = None
    if dim_type == 'Multi':
        open_data = df["Open"].values.reshape((data_len, 1))
        high_data = df["High"].values.reshape((data_len, 1))
        low_data = df["Low"].values.reshape((data_len, 1))
        close_data = df["Close"].values.reshape((data_len, 1))
        marketcap_data = df["Marketcap"].values.reshape((data_len, 1))
        data = np.hstack((close_data, open_data, high_data, low_data, marketcap_data))  # 选取多个维度，并将close_data置于第一列
    else:
        data = df[dim_type].values.reshape((data_len, 1))
    return data[0:int(np.floor(data_len * use_percentage))], np.floor(data_len * use_percentage)

def split_sequence(sequence, dim_type, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of the input pattern
        end_ix = i + n_steps_in
        # find the end of the output pattern
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequence):
            break
        if dim_type == 'Multi':
            # gather input and output parts of the pattern
            seq_x = sequence[i:end_ix, 1:]
            seq_y = sequence[end_ix:out_end_ix, 0]
        else:
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def data_trasform(data, anti=False, scaler=None):
    '''
    MinMax data and anti MinMax data
    :param data: the data source
    :param model: MinMax and anti MinMax
    :param scaler: anti MinMax scaler
    :return: the transformed data
    '''
    if not anti:
        # 归一化
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
    else:
        # 反归一化
        # 如果data是三维数组，去除最后一个维度
        if data.ndim == 3 and data.shape[2] == 1:
            data = data.squeeze(axis=2)

        restored_data = np.zeros_like(data)
        for i in range(data.shape[1]):  # 遍历所有列
            column_data = data[:, i].reshape(-1, 1)
            restored_data[:, i] = scaler.inverse_transform(column_data).ravel()
        return restored_data

def create_model(model_type, n_features, n_steps_in, n_steps_out):
    '''
    create model
    :param model_type:  LSTM,BD LSTM(bidirectional LSTM),ED LSTM(Encoder-Decoder LSTM),CNN
    :param n_features:
    :param n_steps_in:
    :param n_steps_out:
    :return: the created model
    '''
    model = Sequential()
    if model_type == 'LSTM':
        # LSTM
        model.add(LSTM(100, activation='sigmoid', return_sequences=True, input_shape=(n_steps_in, n_features)))
        model.add(LSTM(100, activation='sigmoid'))
        model.add(Dense(n_steps_out))
        model.compile(optimizer='adam', loss='mse')

    elif model_type == 'BD LSTM':
        # bidirectional LSTM
        model.add(Bidirectional(LSTM(100, activation='sigmoid'), input_shape=(n_steps_in, n_features)))
        model.add(Dense(n_steps_out))
        model.compile(optimizer='adam', loss='mse')

    elif model_type == 'ED LSTM':
        # Encoder-Decoder LSTM
        # Encoder
        model.add(LSTM(100, activation='sigmoid', input_shape=(n_steps_in, n_features)))
        # Connector
        model.add(RepeatVector(n_steps_out))
        # Decoder
        model.add(LSTM(100, activation='sigmoid', return_sequences=True))
        model.add(TimeDistributed(Dense(n_steps_out)))
        model.compile(optimizer='adam', loss='mse')

    elif model_type == 'CNN':
        # CNN
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(n_steps_out))
        model.compile(optimizer='adam', loss='mean_squared_error')
    else:
        print("no model")
    return model

def train_and_forecast(model, n_features, dim_type, data_X, data_Y, n_steps_in, n_steps_out, ech):
    X, y = split_sequence(data_X, dim_type, n_steps_in, n_steps_out)

    # 对于多维数据，调整最后一个维度为特征数
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # 训练模型
    model.fit(X, y, epochs=ech, batch_size=32, verbose=1)

    # 拟合模型
    fit_result = []
    for index, ele in enumerate(X):
        print(f'Fitting {index}th data')
        pred = model.predict(ele.reshape((1, n_steps_in, n_features)))
        fit_result.append(pred)
    fr = np.array(fit_result)
    fit_result = fr.reshape(len(fit_result), n_steps_out)

    # 测试模型
    test_x, test_y = split_sequence(data_Y, dim_type, n_steps_in, n_steps_out)
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], n_features))
    test_result = []
    for index, ele in enumerate(test_x):
        print(f'Testing {index}th data')
        pred = model.predict(ele.reshape((1, n_steps_in, n_features)))
        test_result.append(pred)
    tr = np.array(test_result)
    test_result = tr.reshape(len(test_result), n_steps_out)
    return fit_result, test_result

def eval_result(result, n_steps_out, target,mode):
    '''
    evaluate the modl resule
    :param result:the model result
    :param n_steps_out:the days you predict
    :param target:the ground-true
    :param mode:the type of evaluation(you can choose 0：rmse,1：mape)
    :return:the evaluation result
    '''
    if mode==0:
        # return rmse result
        # 归一化
        result, _ = data_trasform(result)
        target, _ = data_trasform(target)
        rmse = []
        for i in range(n_steps_out):
            rmse.append(np.sqrt(np.mean((result[:, i] - target[:, i]) ** 2)))
        return rmse

    elif mode==1:
        # return MAPE result
        result = result + 0.0000001
        target = target + 0.0000001
        mape = []
        for i in range(n_steps_out):
            mape.append(np.mean(np.abs((target[:, i] - result[:, i]) / target[:, i])) * 100)
        return mape
    else:
        return None


def main():
    # -----------------parameters-----------------
    dim_type = 'Close'  # 'Multi' or 'Open', 'High', 'Low', 'Close', 'Marketcap' (选取数据的维度或类型)
    n_steps_in = 5  # 输入步长
    n_steps_out = 1  # 输出步长
    use_percentage = 1  # 使用的数据百分比(=1就是全部数据)
    percentage = 0.7  # 训练集百分比
    epochs = 100  # 迭代次数

    # ---------------get data---------------
    data, data_len=read_data(dim_type, use_percentage)
    # data, scalers = MinMaxdata(data)  # 归一化数据
    data, scalers = data_trasform(data)
    # split into train and test
    data_X = data[0:int(np.floor(data_len * percentage))]  # 训练集
    data_Y = data[int(np.floor(data_len * percentage)):]  # 测试集

    # define the used features
    n_features = 4 if dim_type == 'Multi' else 1

    # ------------------create model and prediction---------------
    model_type = 'ED LSTM' # Encoder-Decoder
    Model=create_model(model_type, n_features, n_steps_in, n_steps_out)
    fit_result, test_result = train_and_forecast(Model, n_features, dim_type, data_X, data_Y, n_steps_in, n_steps_out, epochs)



    # ----------------------evaluation--------------------
    fit_result = data_trasform(fit_result,True, scalers[0]) # 反归一化
    test_result = data_trasform(test_result,True, scalers[0]) # 反归一化

    _, fit = split_sequence(data_X, dim_type, n_steps_in, n_steps_out)
    fit = data_trasform(fit,True, scalers[0]) # 反归一化
    _, test = split_sequence(data_Y, dim_type, n_steps_in, n_steps_out)
    test = data_trasform(test, True, scalers[0]) # 反归一化

    # calc the rmse
    fit_minmax_rmse = eval_result(fit_result,n_steps_out,fit,0)
    test_minmax_rmse = eval_result(test_result, n_steps_out, test, 0)
    print('MINMAX Fit Rmse:', fit_minmax_rmse)
    print('MINMAX Test Rmse:', test_minmax_rmse)


    # calc the MAPE
    fit_MAPE=eval_result(fit_result, n_steps_out, fit, 1)
    test_MAPE = eval_result(test_result, n_steps_out, test, 1)
    print('Fit MAPE:', fit_MAPE)
    print('Test MAPE:', test_MAPE)

    # 绘图
    if n_steps_out == 1:
        # 拟合结果
        plt.figure(figsize=(15, 6))
        plt.plot(fit_result, label='Fit')
        plt.plot(fit, label='Actual')
        plt.title(f'{model_type} Fit Results')
        plt.xlabel('Time Steps')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()

        # 测试结果
        plt.figure(figsize=(15, 6))
        plt.plot(test_result, label='Predicted')
        plt.plot(test, label='Actual')
        plt.title(f'{model_type} Test Results')
        plt.xlabel('Time Steps')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()