# 读取数据
import pandas as pd
import keras
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 使用 pandas 读取CSV文件
df = pd.read_csv(r"coin_Bitcoin.csv")

# 提取 "Open" 列的数据
open_data = df["Open"]
high_data = df["High"]
low_data = df["Low"]
close_data = df["Close"]
marketcap_data = df["Marketcap"]

# 如果需要，将其转换为numpy数组
data_len = open_data.shape[0]
data = open_data.to_numpy().reshape((data_len, 1))
# print(data)

# 初始化归一化器
scaler = MinMaxScaler(feature_range=(0, 1))
# 归一化 open_data
data = scaler.fit_transform(data)


# 按7：3切分数据
data_X = data[0:int(np.floor(len(data) * 0.7))]
data_Y = data[int(np.floor(len(data) * 0.7)):]

n_steps = 3
n_features = 1
n_steps_in = 5
n_steps_out = 3


# split a univariate sequence
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# split a univariate sequence into samples
def multi_split_sequence(sequence, n_steps_in, n_steps_out):
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
    return array(X), array(y)

def train_model_and_forecast(model, data_X, data_Y, n_steps, n_features):
    X, y = split_sequence(data_X, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # 训练模型
    model.fit(X, y, epochs=200, verbose=0)

    # 测试模型
    test_x, test_y = split_sequence(data_Y, n_steps)
    total = len(test_y)
    result = []
    for index, ele in enumerate(test_x):
        # print(f"index:{index} /{total}")
        pred = model.predict(ele.reshape((1, n_steps, n_features)))
        result.append(pred)
    return result


# 保留第一次预测值，剔除掉重复预测值
def multi_train_model_and_forecast_1(model, data_X, data_Y, n_steps_in, n_steps_out, n_features):
    # 训练模型
    X, y = multi_split_sequence(data_X, n_steps_in, n_steps_out)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model.fit(X, y, epochs=200, verbose=0)

    # 测试模型
    test_x, test_y = multi_split_sequence(data_Y, n_steps_in, n_steps_out)
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], n_features))
    result = []
    for index, ele in enumerate(test_x):
        pred = model.predict(ele.reshape((1, n_steps_in, n_features)))

        # 只保留每次预测的第一步结果，除了最后一次迭代
        if index < len(test_x) - 1:
            # 添加第一步预测结果
            result.append(pred[0, 0])
        else:
            # 在最后一次迭代时，添加整个预测结果或其前 n_steps_out 步
            result.extend(pred[0, :n_steps_out])

    return result


# 将重复预测值取平均值
def multi_train_model_and_forecast_2(model, data_X, data_Y, n_steps_in, n_steps_out, n_features):
    X, y = multi_split_sequence(data_X, n_steps_in, n_steps_out)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model.fit(X, y, epochs=200, verbose=0)

    # 测试模型
    test_x, test_y = multi_split_sequence(data_Y, n_steps_in, n_steps_out)
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], n_features))
    result = []
    for index, ele in enumerate(test_x):
        pred = model.predict(ele.reshape((1, n_steps_in, n_features)))

        # 计算每次预测的平均值
        pred_avg = np.mean(pred, axis=1).item() # 使用 .item() 转换为标量

        result.append(pred_avg)

    return result


def calculate_rmse(result, n_steps, data_Y):
    # 检查输入数组长度是否相等
    predicate = np.array(result)
    test_x, test_y = split_sequence(data_Y, n_steps)
    target = test_y
    if len(predicate) != len(target):
        print("预测值和目标值的长度差:", len(predicate) - len(target))
    # 计算RMSE
    rmse = np.sqrt(np.mean((predicate[0:min(len(predicate),len(target))] - target[0:min(len(predicate),len(target))]) ** 2))
    return rmse

# univariate bidirectional lstm example

# 单变量lstm
def create_lstm_model(n_steps, n_features):
    from numpy import array
    from keras.models import Sequential
    from keras.layers import LSTM
    from keras.layers import Dense
    # 定义模型
    model = Sequential()
    model.add(LSTM(100, activation='sigmoid', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(100, activation='sigmoid'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model


def create_bidirectional_lstm_model(n_steps, n_features):
    # 定义模型
    from numpy import array
    from keras.models import Sequential
    from keras.layers import LSTM
    from keras.layers import Dense
    from keras.layers import Bidirectional
    model = Sequential()
    model.add(Bidirectional(LSTM(100, activation='sigmoid'), input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model


def create_ED_lstm_model(n_steps_in, n_steps_out, n_features):
    from numpy import array
    from keras.models import Sequential
    from keras.layers import LSTM
    from keras.layers import Dense
    model = Sequential()
    model.add(LSTM(100, activation='sigmoid', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(100, activation='sigmoid'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')

    return model


lstm_model = create_lstm_model(n_steps, n_features)
bidirectional_lstm_model = create_bidirectional_lstm_model(n_steps, n_features)
ED_lstm_model = create_ED_lstm_model(n_steps_in, n_steps_out, n_features)
# 训练和预测第一个LSTM模型
result_lstm = train_model_and_forecast(lstm_model, data_X, data_Y, n_steps, n_features)
# 训练和预测第二个Bidirectional LSTM模型
result_bidirectional_lstm = train_model_and_forecast(bidirectional_lstm_model, data_X, data_Y, n_steps, n_features)

# 训练和预测第三个ED LSTM模型
result_ED_lstm1 = multi_train_model_and_forecast_1(ED_lstm_model, data_X, data_Y, n_steps_in, n_steps_out, n_features)
result_ED_lstm2 = multi_train_model_and_forecast_2(ED_lstm_model, data_X, data_Y, n_steps_in, n_steps_out, n_features)

# 计算并打印第一个LSTM模型的RMSE
rmse_lstm = calculate_rmse(result_lstm, n_steps, data_Y)
print("LSTM模型的RMSE:", rmse_lstm)

# 计算并打印第二个Bidirectional LSTM模型的RMSE
rmse_bidirectional_lstm = calculate_rmse(result_bidirectional_lstm, n_steps, data_Y)
print("Bidirectional LSTM模型的RMSE:", rmse_bidirectional_lstm)

# 计算并打印第三个ED LSTM模型的RMSE
rmse_ED_lstm1 = calculate_rmse(result_ED_lstm1, n_steps_in, data_Y)
print("ED LSTM模型1的RMSE:", rmse_ED_lstm1)
rmse_ED_lstm2 = calculate_rmse(result_ED_lstm2, n_steps_in, data_Y)
print("ED LSTM模型2的RMSE:", rmse_ED_lstm2)


# 反归一化预测结果
result_lstm = scaler.inverse_transform(np.array(result_lstm).reshape(-1, 1))
result_bidirectional_lstm = scaler.inverse_transform(np.array(result_bidirectional_lstm).reshape(-1, 1))
result_ED_lstm1 = scaler.inverse_transform(np.array(result_ED_lstm1).reshape(-1, 1))
result_ED_lstm2 = scaler.inverse_transform(np.array(result_ED_lstm2).reshape(-1, 1))

# 反归一化测试集的目标值
test_x, test_y = split_sequence(data_Y, n_steps)
target = test_y
target = scaler.inverse_transform(target.reshape(-1, 1))

# 绘制反归一化后的结果
plt.figure(figsize=(12, 6))
plt.plot(result_lstm, label='Lstm', color='navy')
plt.plot(result_bidirectional_lstm, label='Bidirectional_Lstm', color='royalblue')
plt.plot(result_ED_lstm1, label='ED Lstm 1', color='cornflowerblue')
plt.plot(result_ED_lstm2, label='ED Lstm 2', color='lightblue')
plt.plot(target, label='Actual', color='orange')
plt.title('Comparison of Predicted and Actual Open Prices (Original Scale)')
plt.xlabel('Time Steps')
plt.ylabel('Open Price')
plt.legend()
plt.show()









# 计算并打印第三个ED LSTM模型的RMSE
rmse_ED_lstm1 = calculate_rmse(result_ED_lstm1, n_steps_in, data_Y)
print("ED LSTM模型的RMSE:", rmse_ED_lstm1)
rmse_ED_lstm2 = calculate_rmse(result_ED_lstm2, n_steps_in, data_Y)
print("ED LSTM模型的RMSE:", rmse_ED_lstm2)
