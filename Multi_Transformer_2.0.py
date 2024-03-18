#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:39:45 2024

@author: HaochenZhou
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, MultiHeadAttention, Input, LeakyReLU  # 引入LeakyReLU
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2  # 引入L2正则化

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0, reg=0.01):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation=LeakyReLU(), kernel_regularizer=l2(reg))(x)  # 添加LeakyReLU激活函数和L2正则化
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1], kernel_regularizer=l2(reg))(x)  # 添加L2正则化
    return x + res

def build_multivariate_transformer(sequence_length, num_features, head_size=48, num_heads=4, ff_dim=64, num_blocks=4, dropout=0.1):
    inputs = Input(shape=(sequence_length, num_features))
    x = inputs
    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    outputs = Dense(1)(x[:, -1, :])  # Predicting the Close price from the last timestep
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def prepare_data(data, features, target, steps=5):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data[features])

    X, y = [], []
    target_index = features.index(target)

    for i in range(steps, len(scaled_data)):
        X.append(scaled_data[i-steps:i])
        y.append(scaled_data[i, target_index])

    return np.array(X), np.array(y), scaler

def train_and_predict(df, features, target, steps=5):
    X, y, scaler = prepare_data(df, features, target, steps)
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = build_multivariate_transformer(steps, len(features))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    return y_train, y_test, train_predict, test_predict, scaler

def inverse_transform(scaler, predicted_data, original_data, target_index):
    dummy_data = np.zeros_like(original_data)
    dummy_data[:, target_index] = predicted_data.reshape(-1)
    inversed_data = scaler.inverse_transform(dummy_data)
    return inversed_data[:, target_index]

def main():
    df = pd.read_csv('./archive/coin_Bitcoin.csv')
    features = ['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']
    #features = ['Close']
    target = 'Close'
    target_index = features.index(target)
    steps=5
    y_train, y_test, train_predict, test_predict, scaler = train_and_predict(df, features, target, steps)
    
    train_rmse = np.sqrt(np.mean((y_train.flatten()-train_predict.flatten())**2))
    print("Train RMSE:", train_rmse)
    # 计算测试集RMSE
    test_rmse = np.sqrt(np.mean((y_test.flatten()-test_predict.flatten())**2))
    print("Test RMSE:",test_rmse)
    
    # Invert predictions back to original scale
    original_train_data = scaler.transform(df[features].iloc[steps:len(y_train)+steps])
    original_test_data = scaler.transform(df[features].iloc[len(y_train)+steps:len(y_train)+steps+len(y_test)])
    y_train_inv = inverse_transform(scaler, y_train, original_train_data, target_index)
    y_test_inv = inverse_transform(scaler, y_test, original_test_data, target_index)
    train_predict_inv = inverse_transform(scaler, train_predict, original_train_data, target_index)
    test_predict_inv = inverse_transform(scaler, test_predict, original_test_data, target_index)
    
    plt.figure(figsize=(15,6))
    plt.plot(y_train_inv, label='Actual Train')
    plt.plot(train_predict_inv, label='Predicted Train')
    plt.legend()
    plt.title('Training Data: Actual vs Predicted Close Price')
    plt.show()
    
    plt.figure(figsize=(15,6))
    plt.plot(y_test_inv.flatten(), label='Actual Test')
    plt.plot(test_predict_inv.flatten(), label='Predicted Test')
    plt.legend()
    plt.title('Test Data: Actual vs Predicted Close Price')
    plt.show()

if __name__ == '__main__':
    main()
