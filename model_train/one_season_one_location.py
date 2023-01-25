import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.metrics import RootMeanSquaredError
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam
import matplotlib.patches as mpatches
from sklearn.metrics import mean_squared_error as mse
import datetime


# One season 1 location
from model_train.read_data import read_wind_meteodata

df=read_wind_meteodata()
# Ireland
df.index = pd.to_datetime(df['date'], format='%Y.%m.%d %H:%M:%S')
wind = df['wdsp']
# plt.plot(wind)
# plt.show()

def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i + window_size]]
        X.append(row)
        label = df_as_np[i + window_size]
        y.append(label)
    return np.array(X), np.array(y)


def kt_to_ms(speed):
    return speed * 0.514444


def plot_predictions(model_lstm, model_cnn, model_gru, X, y, title='#####', start=0, end=100):
    try:
        print("X[start + 1:end + 1]")
        print(X[start + 1:end + 1])

        # lstm_predictions = model_lstm.predict(X[start + 1:end + 1]).flatten()
        # cnn_predictions = model_cnn.predict(X[start + 1:end + 1]).flatten()
        # gru_predictions = model_gru.predict(X[start + 1:end + 1]).flatten()

        lstm_predictions = model_lstm.predict(X[start:end]).flatten()
        cnn_predictions = model_cnn.predict(X[start:end]).flatten()
        gru_predictions = model_gru.predict(X[start:end]).flatten()

        actual = y[start - 1:end - 1]



        df = pd.DataFrame(data={'LSTM Prediction': lstm_predictions, 'CNN Prediction': cnn_predictions,
                                'GRU Prediction': gru_predictions, 'Actual': actual})

        plt.plot(df['LSTM Prediction'], color='r')
        plt.plot(df['CNN Prediction'], color='b')
        plt.plot(df['GRU Prediction'], color='orange')
        plt.plot(df['Actual'], color='g')
        plt.title(title)
        plt.legend(
            handles=[mpatches.Patch(color='g', label='Actual'), mpatches.Patch(color='r', label='LSTM Prediction'),
                     mpatches.Patch(color='b', label='CNN Prediction'),
                     mpatches.Patch(color='orange', label='GRU Prediction')])
        plt.ylabel('Wind Speed(m/s)')
        plt.xlabel('Hours Ahead')
        plt.show()
        return mse(y[start:end], lstm_predictions), mse(y[start:end], cnn_predictions), mse(y[start:end],
                                                                                            gru_predictions)
    except Exception as e:
        print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
        print(e)


def four_hour_prediction(model, X, y, title, start, end):
    predictions = []
    sequence = []
    input_values = X[start:start+1]
    final_predict_values = [X[start + 1:start + 2]]
    # print("predict_values")
    # print(predict_values)
    actual = y[start - 1:end - 1]
    # [[[5, 2, 4, 1, 6]
    # [2, 4, 1, 6, 5]
    # [4, 1, 6, 5, 3]
    # [1, 6, 5, 3, 4]
    # [6, 5, 3, 4, 5]
    # ]]]
    sequence.append(input_values[0])
    for i in range(0, end - start - 1):
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(input_values)
        l = model.predict(input_values).flatten()
        # l = round(l)
        # input_values = input_values[0][1:]
        # input_values[0].append(l)
        # input_values[0].append(l)
        import random
        # new_l = np.expand_dims(np.array([round(l[0]) + random.choice([4, 3, -3, -4])]), axis=1)
        # print(l[0])
        # print(math.ceil(l[0]))
        wind_speed = math.ceil(l[0]) + 1
        # wind_speed = round(l[0])
        wind_speed = round(l[0])
        # wind_speed = round(l[0] - 0.6)
        

        new_l = np.expand_dims(np.array([wind_speed]), axis=1)

        # new_l = np.expand_dims(np.array([math.ceil(l[0])]), axis=1)
        input_values = np.array(np.append(input_values[0][1:], new_l, axis=0))
        # for k in range(0, len(b) - 1):
        #     print(input_values[0, k+1])
        #     d[k] = [input_values[0, k+1]]
        # d[4] = [l]
        # input_values[0] = d
        # print("input_values")
        # print(input_values)
        sequence.append(input_values)
        input_values = np.expand_dims(input_values, axis=0)
        # print("input_values")
        # print(input_values)

        predictions.append(l)

    # print("predictions")
    # print(predictions)

    # print("*****************************************************************************************")
    # print(predict_values)
    # print(final_predict_values)
    # print(sequence)
    print("*****************************************************************************************")
    print(X[start+1:end+1])
    print(np.array(sequence))
    print("*****************************************************************************************")

    final_sequence = np.array(sequence)
    # np.insert(final_sequence, 0, [X[start:start+1]])
    # final_predictions = model.predict(final_predict_values).flatten()

    final_predictions = model.predict(final_sequence).flatten()

    print("final_predictions")
    print(final_predictions)
    print("actual")
    print(actual)

    df = pd.DataFrame(data={'Prediction': final_predictions, 'Actual': actual})

    plt.plot(df['Prediction'], color='r')
    plt.plot(df['Actual'], color='g')
    plt.title(title)
    plt.legend(handles=[mpatches.Patch(color='g', label='Actual'), mpatches.Patch(color='r', label='Prediction')])
    plt.ylabel('Wind Speed(m/s)')
    plt.xlabel('Hours Ahead')
    plt.show()
    return mse(y[start:end], final_predictions)


WINDOW_SIZE = 5
EPOCHS = 100
TRAIN = False
X1, y1 = df_to_X_y(wind, WINDOW_SIZE)
# print("X1")
# print(X1)
# print("y1")
# print(y1)
# for i in range(sequence_start, 160000):
#     print(X1[i])
#     print(y1[i])
#
# exit(0)
train_len = int(len(X1) * 0.9)
# test_len = int(len(X1) * 0.9)
val_len = int(len(X1) * 1)
print(int(len(X1) * 0.7))
print(int(len(X1) * 0.9))
print(int(len(X1) * 1))
X_train1, y_train1 = X1[:train_len], y1[:train_len]
# X_test1, y_test1 = X1[train_len:test_len], y1[train_len:test_len]
X_val1, y_val1 = X1[train_len:], y1[train_len:]

print(
    "LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM")
print(
    "  LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM   ")
print(
    "LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM")
print(
    "  LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM   ")
print(
    "LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM")
print(
    "  LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM   ")
print(
    "LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM")
# LSTM
# model_lsmt = Sequential()
# model_lsmt.add(InputLayer((5, 1)))
# model_lsmt.add(LSTM(64))
# model_lsmt.add(Dense(8, 'relu'))
# model_lsmt.add(Dense(1, 'linear'))

model_lsmt = Sequential()
model_lsmt.add(LSTM(32, return_sequences=True, activation='sigmoid', input_shape=(X_train1.shape[1], X_train1.shape[2])))
model_lsmt.add(LSTM(units=64, return_sequences=False))
model_lsmt.add(Dense(5))
model_lsmt.add(Dense(1))
model_lsmt.compile(optimizer='adam', loss='mse')

print(model_lsmt.summary())
# Ireland
cp1 = ModelCheckpoint('C:/Users/Maksym_Mysak/PycharmProjects/diploma/LSTM_model_1_L_1_S', save_best_only=True)

# cp1 = ModelCheckpoint('C:/Users/Maksym_Mysak/PycharmProjects/diploma/LSTM_model', save_best_only=True)
# model_lsmt.compile(loss='mse', optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

# if TRAIN:
#     model_lsmt.fit(X_train1, y_train1, batch_size=64, epochs=EPOCHS, shuffle=False, validation_data=(X_val1, y_val1), callbacks=[cp1])
    # model_lsmt.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=EPOCHS, callbacks=[cp1])

print("CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN")
print(
    "  CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN   ")
print("CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN")
print(
    "  CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN   ")
print("CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN")
print(
    "  CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN   ")
print("CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN")
# CNN
model_cnn = Sequential()
model_cnn.add(InputLayer((5, 1)))
model_cnn.add(Conv1D(64, kernel_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(8, 'relu'))
model_cnn.add(Dense(1, 'linear'))
# model_cnn = Sequential()
# model_cnn.add(Conv1D(32,  kernel_size=2, activation='sigmoid', input_shape=(X_train1.shape[1], X_train1.shape[2])))
# model_cnn.add(Conv1D(64, kernel_size=2))
# model_cnn.add(Dense(5))
# model_cnn.add(Dense(1))
model_cnn.compile(optimizer='adam', loss='mse')
print(model_cnn.summary())

# Ireland
cp2 = ModelCheckpoint('C:/Users/Maksym_Mysak/PycharmProjects/diploma/CNN_model_1_L_1_S', save_best_only=True)

# cp2 = ModelCheckpoint('C:/Users/Maksym_Mysak/PycharmProjects/diploma/CNN_model', save_best_only=True)
# model_cnn.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

if TRAIN:
    model_cnn.fit(X_train1, y_train1, batch_size=64, epochs=EPOCHS, shuffle=False, validation_data=(X_val1, y_val1), callbacks=[cp2])

print("GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU")
print(
    "  GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU   ")
print("GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU")
print(
    "  GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU   ")
print("GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU")
print(
    "  GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU   ")
print("GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU")

# GRU
# model_gru = Sequential()
# model_gru.add(InputLayer((5, 1)))
# model_gru.add(GRU(64))
# model_gru.add(Dense(8, 'relu'))
# model_gru.add(Dense(1, 'linear'))
model_gru = Sequential()
model_gru.add(GRU(32, return_sequences=True, activation='sigmoid', input_shape=(X_train1.shape[1], X_train1.shape[2])))
model_gru.add(GRU(units=64, return_sequences=False))
model_gru.add(Dense(5))
model_gru.add(Dense(1))
model_gru.compile(optimizer='adam', loss='mse')
model_gru.summary()

# Ireland
cp3 = ModelCheckpoint('C:/Users/Maksym_Mysak/PycharmProjects/diploma/GRU_model_1_L_1_S', save_best_only=True)

# cp3 = ModelCheckpoint('C:/Users/Maksym_Mysak/PycharmProjects/diploma/GRU_model', save_best_only=True)
# model_gru.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

# if TRAIN:
#     model_gru.fit(X_train1, y_train1, batch_size=64, epochs=EPOCHS, shuffle=False, validation_data=(X_val1, y_val1), callbacks=[cp3])

# Ireland
model_lsmt = load_model('C:/Users/Maksym_Mysak/PycharmProjects/diploma/LSTM_model_1_L_1_S')
model_cnn = load_model('C:/Users/Maksym_Mysak/PycharmProjects/diploma/CNN_model_1_L_1_S')
model_gru = load_model('C:/Users/Maksym_Mysak/PycharmProjects/diploma/GRU_model_1_L_1_S')

# print("Test 1")
# print(plot_predictions(model_lsmt, model_cnn, model_gru, X_train1, y_train1, 'Test 1', 1340, 1360))
# print("Test 2")
# print(plot_predictions(model_lsmt, model_cnn, model_gru, X_train1, y_train1, 'Test 2', 1380, 1400))
#
# print("Test 3")
# print(plot_predictions(model_lsmt, model_cnn, model_gru, X_train1, y_train1, 'Test 3', 1660, 1690))
#
# print("Test 4")
# print(plot_predictions(model_lsmt, model_cnn, model_gru, X_train1, y_train1, 'Test 4', 1810, 1815))
# print(four_hour_prediction(model_lsmt, X_train1, y_train1, 'Pure LSTM prediction', 1810, 1815))
# print(four_hour_prediction(model_cnn, X_train1, y_train1, 'Pure CNN prediction', 1810, 1815))
# print(four_hour_prediction(model_gru, X_train1, y_train1, 'Pure GRU prediction', 1810, 1815))
# print("Test 5")
# print(plot_predictions(model_lsmt, model_cnn, model_gru, X_train1, y_train1, 'Test 5', 1840, 1860))
#
# print("Test 6")
# print(plot_predictions(model_lsmt, model_cnn, model_gru, X_train1, y_train1, 'Test 6', 3020, 3060))
#
# print("Test 7")
# print(plot_predictions(model_lsmt, model_cnn, model_gru, X_train1, y_train1, 'Test 7', 4240, 4270))
# print("Test 8")
# print(plot_predictions(model_lsmt, model_cnn, model_gru, X_train1, y_train1, 'Test 8', 4340, 4370))
#
# print("Test 9")
# print(plot_predictions(model_lsmt, model_cnn, model_gru, X_train1, y_train1, 'Test 9', 4355, 4375))
#
# print("Test 10")
# print(plot_predictions(model_lsmt, model_cnn, model_gru, X_train1, y_train1, 'Test 10', 4935, 4955))
# print("Test 11")
# print(plot_predictions(model_lsmt, model_cnn, model_gru, X_train1, y_train1, 'Test 11', 60, 64))
# print(four_hour_prediction(model_lsmt, X_train1, y_train1, 'Pure LSTM prediction', 60, 64))
# print(four_hour_prediction(model_cnn, X_train1, y_train1, 'Pure CNN prediction', 60, 64))
# print(four_hour_prediction(model_gru, X_train1, y_train1, 'Pure GRU prediction', 60, 64))

# print("Test 12")
# print(plot_predictions(model_lsmt, model_cnn, model_gru, X_train1, y_train1, 'Test 12', 90, 94))
# print(four_hour_prediction(model_lsmt, X_train1, y_train1, 'Pure LSTM prediction', 90, 94))
# print(four_hour_prediction(model_cnn, X_train1, y_train1, 'Pure CNN prediction', 90, 94))
# print(four_hour_prediction(model_gru, X_train1, y_train1, 'Pure GRU prediction', 90, 94))

# print("Test 13")
# print(plot_predictions(model_lsmt, model_cnn, model_gru, X_train1, y_train1, 'Test 13', 1100, 1104))
# print(four_hour_prediction(model_lsmt, X_train1, y_train1, 'Pure LSTM prediction', 1100, 1104))
# print(four_hour_prediction(model_cnn, X_train1, y_train1, 'Pure CNN prediction', 1100, 1104))
# print(four_hour_prediction(model_gru, X_train1, y_train1, 'Pure GRU prediction', 1100, 1104))

# print("Test 14")
print(plot_predictions(model_lsmt, model_cnn, model_gru, X_val1, y_val1, 'Test 14', 1130, 1134))
print(four_hour_prediction(model_lsmt, X_val1, y_val1, 'Pure LSTM prediction', 1130, 1134))
print(four_hour_prediction(model_cnn, X_val1, y_val1, 'Pure CNN prediction', 1130, 1134))
print(four_hour_prediction(model_gru, X_val1, y_val1, 'Pure GRU prediction', 1130, 1134))

# print(plot_predictions(model_lsmt, model_cnn, model_gru, X_train1, y_train1, 'To comp 1', 1340, 1344))
# print(four_hour_prediction(model_lsmt, X_train1, y_train1, 'Pure LSTM prediction', 1340, 1345))
# print(four_hour_prediction(model_cnn, X_train1, y_train1, 'Pure CNN prediction', 1340, 1344))
# print(four_hour_prediction(model_gru, X_train1, y_train1, 'Pure GRU prediction', 1341, 1345))

# print(plot_predictions(model_lsmt, model_cnn, model_gru, X_train1, y_train1, 'To comp 2', 1376, 1380))
# print(four_hour_prediction(model_lsmt, X_train1, y_train1, 'Pure LSTM prediction', 1376, 1380))
# print(four_hour_prediction(model_cnn, X_train1, y_train1, 'Pure CNN prediction', 1376, 1380))
# print(four_hour_prediction(model_gru, X_train1, y_train1, 'Pure GRU prediction', 1376, 1380))

end = datetime.datetime.now()

print(start)
print(end)
