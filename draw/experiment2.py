import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Conv1D, Flatten, MaxPooling1D, RepeatVector, \
    TimeDistributed, LayerNormalization, Dropout, MultiHeadAttention, Input
from tensorflow.keras.optimizers import Adam

matplotlib.rcParams['font.family'] = 'Times New Roman'
pd.set_option('mode.chained_assignment', None)


def read_data(path, features):
    data = pd.read_csv(path)
    data["Date"] = pd.to_datetime(data["Date"], format='%Y-%m-%d').dt.date
    data.set_index('Date', inplace=True)
    data.sort_index(ascending=True, inplace=True)  # 时间升序排列，旧日期在前面
    data = data[features]
    return data


def get_model(model_name, n_steps_in, n_steps_out):
    model = Sequential()
    adam_optimizer = Adam(learning_rate=0.001)
    if model_name == 'BD LSTM':
        # mymodel.add(keras.Input(shape=(n_steps_in,50)))
        model.add(Bidirectional(LSTM(50, activation='sigmoid'), input_shape=(n_steps_in, 1)))
        model.add(Dense(n_steps_out))
    elif model_name == 'ED LSTM':
        # Encoder-Decoder LSTM
        # Encoder
        print("here")
        model.add(LSTM(100, activation='sigmoid', input_shape=(n_steps_in, 1)))
        # Connector
        model.add(RepeatVector(n_steps_out))
        # Decoder
        model.add(LSTM(100, activation='sigmoid', return_sequences=True))
        model.add(TimeDistributed(Dense(1)))
    else:
        raise Exception("未找到模型")
    model.compile(optimizer=adam_optimizer,
                  loss='mse',
                  metrics=[
                      keras.metrics.RootMeanSquaredError(),
                      keras.metrics.MeanAbsolutePercentageError()
                  ])
    return model


def data_trasform(data, anti=False, scaler=None):
    # data为numpy数据
    if not anti:
        scaler = MinMaxScaler()
        data_norm = scaler.fit_transform(data)
        return data_norm, scaler
    else:
        return scaler.inverse_transform(data)


def draw(data, save_name, step, confidence):
    # 选择与第一步骤相关的列
    pred_columns = [col for col in data.columns if col.endswith(f'_step{step}')]
    data_pred = data[pred_columns]

    # n
    n = data_pred.shape[1]
    # mean
    pred_mean = data_pred.mean(axis=1)
    # std
    pred_std = data_pred.std(axis=1)

    # 计算high、low
    t_critical = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    high = pred_mean + (t_critical * pred_std / np.sqrt(n))
    low = pred_mean - (t_critical * pred_std / np.sqrt(n))

    data['Prediction_step' + str(step)] = pred_mean
    data['High_step' + str(step)] = high
    data['Low_step' + str(step)] = low

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data.index, data['Actual'], color='orange')
    ax.plot(data.index, data['Prediction_step' + str(step)], color='blue')
    ax.fill_between(data.index, data['Low_step' + str(step)], data['High_step' + str(step)], color='gray', alpha=0.5)
    ax.legend(['Actual', f'Prediction Step {step}', 'Uncertainty'])
    ax.tick_params(axis='x', labelrotation=45)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Close Price (USD)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_name, dpi=600)
    plt.show()
    return data


# 计算每个步骤的置信区间
def calculate_stepwise_confidence_intervals(df, n_steps_out, confidence=0.95):
    intervals_df = pd.DataFrame()
    for step in range(1, n_steps_out + 1):
        for metric in ['Train_RMSE', 'Test_RMSE', 'Train_MAPE', 'Test_MAPE']:
            values = df[f'{metric}_Step{step}']
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            n = len(values)
            t_critical = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
            margin_of_error = t_critical * std / np.sqrt(n)
            intervals_df.loc[f'{metric}_Step{step}', 'Lower_Bound'] = mean - margin_of_error
            intervals_df.loc[f'{metric}_Step{step}', 'Upper_Bound'] = mean + margin_of_error
    return intervals_df


def eval_result(result, n_steps_out, target, mode):
    '''
    evaluate the modl resule
    :param result:the model result
    :param n_steps_out:the days you predict
    :param target:the ground-true
    :param mode:the type of evaluation(you can choose 0：rmse,1：mape)
    :return:the evaluation result
    '''
    if mode == 0:
        # return rmse result
        # 归一化
        result, _ = data_trasform(result)
        target, _ = data_trasform(target)
        # 下面需要修改
        rmse = []
        for i in range(n_steps_out):
            rmse.append(np.sqrt(np.mean((result[:, i] - target[:, i]) ** 2)))
        return rmse

    elif mode == 1:
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
    # 初始化参数
    n_steps_in = 3
    n_features = 1
    n_steps_out = 1
    batch_size = 32
    epochs = 100  # epoch大小
    shuffle = True  # 是否shuffle
    rounds = 30  # 实验次数
    verbose = 0

    # 获取模型
    model_name = 'BD LSTM'
    model = get_model(model_name, n_steps_in, n_steps_out)

    model_path = Path("model_hub")
    if not model_path.exists():
        model_path.mkdir()

    # 读取数据
    file_path = r"./BTC-USD2019.csv"
    features = ['Close']
    data = read_data(file_path, features)

    # 归一化
    norm_values, scaler = data_trasform(data.values)
    data_norm = pd.DataFrame(data=norm_values, index=data.index, columns=data.columns, copy=True)

    # 划分数据集
    ratio = 0.7
    data_len = data_norm.shape[0]
    train_set, test_set = data_norm.iloc[0:int(data_len * ratio), ], data_norm.iloc[int(data_len * ratio):, ]

    y_train = keras.utils.timeseries_dataset_from_array(train_set.values[n_steps_in:],
                                                        targets=None,
                                                        sequence_length=n_steps_out,
                                                        batch_size=None)

    x_train = keras.utils.timeseries_dataset_from_array(train_set.values[:-n_steps_out],
                                                        targets=None,
                                                        sequence_length=n_steps_in,
                                                        batch_size=None)

    y_test = keras.utils.timeseries_dataset_from_array(test_set.values[n_steps_in:],
                                                       targets=None,
                                                       sequence_length=n_steps_out,
                                                       batch_size=None)

    x_test = keras.utils.timeseries_dataset_from_array(test_set.values[:-n_steps_out],
                                                       targets=None,
                                                       sequence_length=n_steps_in,
                                                       batch_size=None)

    x_train = np.array(list(x_train.as_numpy_iterator()))
    y_train = np.array(list(y_train.as_numpy_iterator()))
    x_test = np.array(list(x_test.as_numpy_iterator()))
    y_test = np.array(list(y_test.as_numpy_iterator()))

    # 训练
    result = test_set.iloc[n_steps_in + n_steps_out - 1:]
    result.rename(columns={'Close': "Actual"}, inplace=True)
    # 反归一化
    result['Actual'] = data_trasform(result['Actual'].values.reshape(-1, 1), anti=True, scaler=scaler)

    train_history = None
    exp_result = pd.DataFrame({}, columns=['Train MINMAX RMSE', 'Test MINMAX RMSE', 'Train MAPE', 'Test MAPE'])
    # 收集评估指标
    columns = ['Round']
    for step in range(1, n_steps_out + 1):
        columns.extend(
            [f'Train_RMSE_Step{step}', f'Test_RMSE_Step{step}', f'Train_MAPE_Step{step}', f'Test_MAPE_Step{step}'])
    metrics_df = pd.DataFrame(columns=columns)
    for round in range(rounds):
        print(f"Training: {round}/{rounds}")
        column_name = 'round' + str(round)

        # 保存模型
        model_save_name = Path(model_path, (model_name + str(round) + ".keras"))

        # 训练模型
        train_history = model.fit(x_train, y_train, batch_size=batch_size, shuffle=shuffle, epochs=epochs,
                                  verbose=verbose)
        x_train_pred = model.predict(x_train, batch_size=batch_size, verbose=verbose)
        x_test_pred = model.predict(x_test, batch_size=batch_size, verbose=verbose)

        model.save(model_save_name)
        # 保存预测结果
        for i in range(n_steps_out):
            column_name = f'round{round}_step{i + 1}'
            result[column_name] = data_trasform(x_test_pred[:, i].reshape(-1, 1), anti=True, scaler=scaler)
        result.to_excel(model_name + " result.xlsx")

        # 反归一化预测结果和实际数据
        x_train_pred_unnorm = data_trasform(x_train_pred.reshape(-1, n_steps_out), anti=True, scaler=scaler)
        y_train_unnorm = data_trasform(y_train.reshape(-1, n_steps_out), anti=True, scaler=scaler)
        x_test_pred_unnorm = data_trasform(x_test_pred.reshape(-1, n_steps_out), anti=True, scaler=scaler)
        y_test_unnorm = data_trasform(y_test.reshape(-1, n_steps_out), anti=True, scaler=scaler)

        # 计算并收集指标
        metrics_row = [round]
        train_rmse = eval_result(x_train_pred_unnorm, n_steps_out, y_train_unnorm, 0)
        test_rmse = eval_result(x_test_pred_unnorm, n_steps_out, y_test_unnorm, 0)
        train_mape = eval_result(x_train_pred_unnorm, n_steps_out, y_train_unnorm, 1)
        test_mape = eval_result(x_test_pred_unnorm, n_steps_out, y_test_unnorm, 1)
        for step in range(n_steps_out):
            metrics_row.extend([train_rmse[step], test_rmse[step], train_mape[step], test_mape[step]])
        metrics_df.loc[len(metrics_df)] = metrics_row

        # 保存评估结果
        exp_result.loc[len(exp_result.index)] = [train_rmse,
                                                 test_rmse,
                                                 train_mape,
                                                 test_mape]
        exp_result.to_excel(model_name + 'RMSEMAPE.xlsx')

    # 保存训练历史
    history = pd.DataFrame(train_history.history)
    model_history_path = model_name + " train history.xlsx"
    history.to_excel(model_history_path)

    # 绘图，专注于第一步
    fig_save_name = model_name + ' Prediction.png'
    data_draw = draw(result, fig_save_name, step=1, confidence=0.9)
    data_draw.to_excel(model_name + " draw data.xlsx")

    confidence_intervals_df = calculate_stepwise_confidence_intervals(metrics_df, n_steps_out, confidence=0.95)
    confidence_intervals_df.to_excel(model_name + '_ConfidenceIntervals.xlsx')


if __name__ == '__main__':
    main()


