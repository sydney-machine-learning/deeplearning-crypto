# 读取数据
import pandas as pd
import keras
import numpy as np
from numpy import array

# 使用 pandas 读取CSV文件
df = pd.read_csv(r"C:\Users\Administrator\Desktop\新建文件夹\coin_Bitcoin.csv")

# 提取 "Open" 列的数据
open_data = df["Open"]
high_data = df["High"]
low_data = df["Low"]
close_data = df["Close"]
marketcap_data = df["Marketcap"]

# 如果需要，将其转换为numpy数组
data = open_data.to_numpy()
# print(data)

# 按7：3切分数据
data_X = data[0:int(np.floor(len(data) * 0.7))]
data_Y = data[int(np.floor(len(data) * 0.7)):]

n_steps = 3
n_features = 1
n_steps_in = 3
n_steps_out = 2


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

def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def train_model_and_forecast(model, data_X, data_Y, n_steps, n_features):
    X, y = split_sequence(data_X, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # 训练模型
    model.fit(X, y, epochs=200, verbose=0)

    result = X[-1].flatten().tolist()

    for i in range(len(data_Y)):
        x_input = np.array(result[i:i + n_steps])
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        result.append(*yhat.ravel().tolist())

    return result


# 保留第一次预测值，剔除掉重复预测值
def multi_train_model_and_forecast_1(model, data_X, data_Y, n_steps_in, n_steps_out, n_features):
    X, y = multi_split_sequence(data_X, n_steps_in, n_steps_out)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model.fit(X, y, epochs=200, verbose=0)
    result = X[-1].flatten().tolist()

    for i in range(len(data_Y) - n_steps_out + 1):
        x_input = np.array(result[i:i + n_steps_in])
        x_input = x_input.reshape((1, n_steps_in, n_features))
        yhat = model.predict(x_input, verbose=0)
        if i == 1:
            result.extend(yhat.ravel().tolist())
        else:
            result.append(yhat[0][-1])
    return result


# 将重复预测值取平均值
def multi_train_model_and_forecast_2(model, data_X, data_Y, n_steps_in, n_steps_out, n_features):
    X, y = multi_split_sequence(data_X, n_steps_in, n_steps_out)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model.fit(X, y, epochs=200, verbose=0)
    result = X[-1].flatten().tolist()

    for i in range(len(data_Y) - n_steps_out + 1):
        x_input = np.array(result[i:i + n_steps_in])
        x_input = x_input.reshape((1, n_steps_in, n_features))
        yhat = model.predict(x_input, verbose=0)
        if i == 1:
            result.extend(yhat.ravel().tolist())
        else:
            for j in range(n_steps_out-1):
                result[-j] = (result[-j] + yhat[0][j-1]) / 2
            result.append(yhat[0][-1])
        return result

def calculate_rmse(result, n_steps, data_Y):
    # 检查输入数组长度是否相等

    predicate = np.array(result[n_steps:])
    target = data_Y

    if len(predicate) != len(target):
        print("预测值和目标值的长度差:", len(predicate) - len(target))

    # 计算RMSE
    rmse = np.sqrt(np.mean((predicate - target) ** 2))

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
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
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
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model


def create_ED_lstm_model(n_steps_in, n_steps_out, n_features):
    from numpy import array
    from keras.models import Sequential
    from keras.layers import LSTM
    from keras.layers import Dense
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(50, activation='relu'))
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
print("ED LSTM模型的RMSE:", rmse_ED_lstm1)
rmse_ED_lstm2 = calculate_rmse(result_ED_lstm2, n_steps_in, data_Y)
print("ED LSTM模型的RMSE:", rmse_ED_lstm2)
