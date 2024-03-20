# 读取数据
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Conv1D, Flatten, MaxPooling1D, RepeatVector, \
    TimeDistributed, LayerNormalization, Dropout, MultiHeadAttention, Input
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt




def read_data(path, dim_type, gold_path=None, use_percentage=1):
    df = pd.read_csv(path)
    data_len = df.shape[0]
    data = None
    if dim_type!='Multi':
        data = df[dim_type].values.reshape((data_len, 1))
    else:
        # Multi
        df["Date"]=pd.to_datetime(df["Date"], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d')
        open_data = df["Open"].values.reshape((data_len, 1))
        high_data = df["High"].values.reshape((data_len, 1))
        low_data = df["Low"].values.reshape((data_len, 1))
        close_data = df["Close"].values.reshape((data_len, 1))
        if gold_path is not None:
            gold=pd.read_excel(gold_path) # 读取金价
            gold['Date']=gold['Date'].dt.strftime('%Y-%m-%d')
            df['Gold']=df['Date']
            
            # calc the gold series
            df['Gold'] = df['Gold'].apply(lambda x: gold['Price'][x==gold['Date']].values[0] if x in gold['Date'].values else np.nan)  
            # fillna using interpolating
            df['Gold'] = df['Gold'].interpolate(limit_direction="both")

            gold_data=df["Gold"].values.reshape((data_len, 1))
            data = np.hstack((close_data, open_data, high_data, low_data, gold_data))
        else:
            data = np.hstack((close_data, open_data, high_data, low_data))
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
        scalers = {}
        normalized_data = np.zeros_like(data)
        for i in range(data.shape[1]):  # data.shape[1] 是列的数量
            scaler = MinMaxScaler()
            column_data = data[:, i].reshape(-1, 1)
            normalized_column = scaler.fit_transform(column_data)
            normalized_data[:, i] = normalized_column.ravel()
            scalers[i] = scaler
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
        
        

def create_transformer_model(input_seq_length, output_seq_length, num_features, d_model, num_heads, ff_dim,
                             num_transformer_blocks, dropout_rate=0.1):
    inputs = Input(shape=(input_seq_length, num_features))

    x = Dense(d_model)(inputs)

    for _ in range(num_transformer_blocks):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)  # Dropout after attention
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)

        ff_output = Dense(ff_dim, activation="relu")(x)
        ff_output = Dropout(dropout_rate)(ff_output)  # Dropout after first dense layer
        ff_output = Dense(d_model)(ff_output)

        x = LayerNormalization(epsilon=1e-6)(x + ff_output)

    outputs = Dense(output_seq_length)(x[:, -1, :])  # We take the last step's output for forecasting
    model = Model(inputs, outputs)
    return model


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
    adam_optimizer = Adam(learning_rate=0.001)
    if model_type == 'LSTM':
        # LSTM
        model.add(LSTM(100, activation='sigmoid', return_sequences=True, input_shape=(n_steps_in, n_features)))
        model.add(LSTM(100, activation='sigmoid'))
        model.add(Dense(n_steps_out))

    elif model_type == 'BD LSTM':
        # bidirectional LSTM
        model.add(Bidirectional(LSTM(50, activation='sigmoid'), input_shape=(n_steps_in, n_features)))
        model.add(Dense(n_steps_out))

    elif model_type == 'ED LSTM':
        # Encoder-Decoder LSTM
        # Encoder
        model.add(LSTM(100, activation='sigmoid', input_shape=(n_steps_in, n_features)))
        # Connector
        model.add(RepeatVector(n_steps_out))
        # Decoder
        model.add(LSTM(100, activation='sigmoid', return_sequences=True))
        model.add(TimeDistributed(Dense(1)))

    elif model_type == 'CNN':
        # CNN
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(20, activation='relu'))
        model.add(Dense(n_steps_out))

    elif model_type == 'Convolutional LSTM':
        # Convolutional LSTM
        model.add(Conv1D(filters=20, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
        model.add(LSTM(20, activation='relu', return_sequences=False))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(n_steps_out))

    elif model_type == 'Transformer':
        model = create_transformer_model(n_steps_in, n_steps_out, n_features, d_model=64,
                                         num_heads=12, ff_dim=64, num_transformer_blocks=3)

    else:
        print("no model")
    model.compile(optimizer=adam_optimizer, loss='mse')
    return model


def train_and_forecast(model, n_features, dim_type, data_X, data_Y, n_steps_in, n_steps_out, ech):
    # 训练模型
    # 隐藏输出
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    X, y = split_sequence(data_X, dim_type, n_steps_in, n_steps_out)
    # 对于多维数据，调整最后一个维度为特征数
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model.fit(X, y, epochs=ech, batch_size=32, verbose=1)

    # 拟合结果
    fit_result = []
    for index, ele in enumerate(X):
        print(f'Fitting {index}th data')
        pred = model.predict(ele.reshape((1, n_steps_in, n_features)))
        fit_result.append(pred)
    fr = np.array(fit_result)
    fit_result = fr.reshape(len(fit_result), n_steps_out)
    # 测试结果
    test_x, test_y = split_sequence(data_Y, dim_type, n_steps_in, n_steps_out)
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], n_features))
    test_result = []
    for index, ele in enumerate(test_x):
        print(f'Predicting {index}th data')
        pred = model.predict(ele.reshape((1, n_steps_in, n_features)))
        test_result.append(pred)
    tr = np.array(test_result)
    test_result = tr.reshape(len(test_result), n_steps_out)

    # 恢复输出
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return fit_result, test_result

def eval_result(result, n_steps_out, target, mode):
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
        # 下面需要修改
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
    print("start")
    # -----------------parameters-----------------
    model_hub = ['LSTM', 'BD LSTM', 'ED LSTM', 'CNN', 'Convolutional LSTM', 'Transformer']
    file_path = r"./coin_Bitcoin.csv"
    dim_type = 'Multi'  # 'Multi' or 'Open', 'High', 'Low', 'Close'(选取数据的维度或类型)
    gold_path = r"./GoldPrice.xlsx"
    use_percentage = 1  # 使用的数据百分比(=1就是全部数据)

    n_steps_in = 6  # 输入步长
    n_steps_out = 5 # 输出步长

    percentage = 0.7  # 训练集百分比
    epochs = 200  # 迭代次数
    rounds=2  # Number of exp

    # ---------------get data---------------

    data, data_len = read_data(file_path, dim_type, gold_path, use_percentage)
    # data, scalers = MinMaxdata(data)  # 归一化数据
    data, scalers = data_trasform(data)
    # split into train and test
    train_set = data[0:int(np.floor(data_len * percentage))]  # 训练集
    test_set = data[int(np.floor(data_len * percentage)):]  # 测试集

    # ----------保存、重新读取数据集

    # -------------------------

    # define the used features
    # used gold price or no used
    n_features = len(train_set[0]) - 1 if len(train_set[0]) > 1 else 1

    # ------------------create model and prediction---------------
    model_type = 'Transformer' # Encoder-Decoder
    Model = create_model(model_type, n_features, n_steps_in, n_steps_out)

    exp_result=pd.DataFrame({},columns=['Train MINMAX RMSE', 'Test MINMAX RMSE', 'Train MAPE', 'Test MAPE'])
    

    for round in range(rounds):
        print(f"the {round}-th exp, total:{rounds} rounds")

        
        train_result, test_result = train_and_forecast(Model, n_features, dim_type, train_set, test_set, n_steps_in,
                                                     n_steps_out, epochs)

        # ----------------------evaluation--------------------
        train_result = data_trasform(train_result, True, scalers[0])  # 反归一化
        test_result = data_trasform(test_result, True, scalers[0])  # 反归一化

        _, train = split_sequence(train_set, dim_type, n_steps_in, n_steps_out)
        train = data_trasform(train, True, scalers[0])  # 反归一化
        _, test = split_sequence(test_set, dim_type, n_steps_in, n_steps_out)
        test = data_trasform(test, True, scalers[0])  # 反归一化
        

        # calc the rmse  
        train_minmax_rmse = eval_result(train_result, n_steps_out, train, 0)
        test_minmax_rmse = eval_result(test_result, n_steps_out, test, 0)
        
        print('Train MINMAX RMSE:', train_minmax_rmse)
        print('Test MINMAX RMSE:', test_minmax_rmse)

        # calc the MAPE
        train_MAPE = eval_result(train_result, n_steps_out, train, 1)
        test_MAPE = eval_result(test_result, n_steps_out, test, 1)
        print('Train MAPE:', train_MAPE)
        print('Test MAPE:', test_MAPE)

        # save result
        exp_result.loc[len(exp_result.index)] = [train_minmax_rmse,
                                                  test_minmax_rmse,
                                                  train_MAPE,
                                                  test_MAPE]
        exp_result.to_excel("exp_results.xlsx")


    # 绘图
    if n_steps_out == 1:
        # 拟合结果
        plt.figure(figsize=(15, 6))
        plt.plot(train_result, label='Train')
        plt.plot(train, label='Actual')
        plt.title(f'{model_type} Train Results')
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

