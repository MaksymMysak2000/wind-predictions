from model_train.read_data import read_wind_meteodata
from model_train.read_data import df_to_X_y
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.callbacks import ModelCheckpoint


EPOCHS = 50
TRAIN_LSTM = False
TRAIN_CNN = False
TRAIN_GRU = False
WINDOW_SIZE = 5


def train_models():

    wind = read_wind_meteodata()

    # prepare input/output values
    X1, y1 = df_to_X_y(wind, WINDOW_SIZE)

    train_len = int(len(X1) * 0.7)
    test_len = int(len(X1) * 0.9)
    val_len = int(len(X1) * 1)

    # split data to train, test and validations arrays
    X_test1, y_test1 = X1[train_len:test_len], y1[train_len:test_len]
    X_train1, y_train1 = X1[:train_len], y1[:train_len]
    X_val1, y_val1 = X1[train_len:], y1[train_len:]

    print(
        "LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM")
    print(
        "  LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM   ")

    # lstm
    model_lsmt = Sequential()
    model_lsmt.add(
        LSTM(32, return_sequences=True, activation='sigmoid', input_shape=(X_train1.shape[1], X_train1.shape[2])))
    model_lsmt.add(LSTM(units=64, return_sequences=False))
    model_lsmt.add(Dense(5))
    model_lsmt.add(Dense(1))
    model_lsmt.compile(optimizer='adam', loss='mse')

    print(model_lsmt.summary())

    m_ch_lstm = ModelCheckpoint('C:/Users/Maksym_Mysak/PycharmProjects/wind-prediction/prediction_model/lstm/LSTM_model_1_L_1_S', save_best_only=True)

    if TRAIN_LSTM:
        model_lsmt.fit(X_train1, y_train1, batch_size=64, epochs=EPOCHS, shuffle=False, validation_data=(X_val1, y_val1),
                   callbacks=[m_ch_lstm])


    print(
        "  CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN   ")
    print(
        "CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN")

    # CNN
    model_cnn = Sequential()
    model_cnn.add(InputLayer((5, 1)))
    model_cnn.add(Conv1D(64, kernel_size=2))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(8, 'relu'))
    model_cnn.add(Dense(1, 'linear'))
    model_cnn.compile(optimizer='adam', loss='mse')

    print(model_cnn.summary())

    m_ch_cnn = ModelCheckpoint('C:/Users/Maksym_Mysak/PycharmProjects/wind-prediction/prediction_model/cnn/CNN_model_1_L_1_S', save_best_only=True)

    if TRAIN_CNN:
        model_cnn.fit(X_train1, y_train1, batch_size=64, epochs=EPOCHS, shuffle=False, validation_data=(X_val1, y_val1),
                  callbacks=[m_ch_cnn])

    print(
        "  GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU   ")
    print(
        "GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU GRU")


    # GRU
    model_gru = Sequential()
    model_gru.add(
        GRU(32, return_sequences=True, activation='sigmoid', input_shape=(X_train1.shape[1], X_train1.shape[2])))
    model_gru.add(GRU(units=64, return_sequences=False))
    model_gru.add(Dense(5))
    model_gru.add(Dense(1))
    model_gru.compile(optimizer='adam', loss='mse')
    model_gru.summary()

    m_ch_gru = ModelCheckpoint('C:/Users/Maksym_Mysak/PycharmProjects/wind-prediction/prediction_model/gru/GRU_model_1_L_1_S', save_best_only=True)

    if TRAIN_GRU:
        model_gru.fit(X_train1, y_train1, batch_size=64, epochs=EPOCHS, shuffle=False, validation_data=(X_val1, y_val1), callbacks=[m_ch_gru])
