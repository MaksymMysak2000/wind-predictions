import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
import numpy as np
import pandas as pd


def four_hour_prediction(model, X, y, title, start, end):
    predictions = []
    sequence = []
    input_values = X[start:start + 1]

    actual = y[start:end]
    # [[[5, 2, 4, 1, 6]
    # [2, 4, 1, 6, 5]
    # [4, 1, 6, 5, 3]
    # [1, 6, 5, 3, 4]
    # [6, 5, 3, 4, 5]
    # ]]]
    sequence.append(input_values[0])

    for i in range(0, end - start - 1):
        l = model.predict(input_values).flatten()

        wind_speed = math.ceil(l[0])

        new_l = np.expand_dims(np.array([wind_speed]), axis=1)

        input_values = np.array(np.append(input_values[0][1:], new_l, axis=0))

        sequence.append(input_values)
        input_values = np.expand_dims(input_values, axis=0)

        predictions.append(l)

    final_sequence = np.array(sequence)
    final_predictions = model.predict(final_sequence).flatten()

    final_predictions_m_per_s = [i * 0.511111 for i in final_predictions]
    actual_m_per_s = [i * 0.511111 for i in actual]

    df = pd.DataFrame(data={'Prediction': final_predictions_m_per_s, 'Actual': actual_m_per_s})
    df.index = np.arange(1, len(df) + 1)

    plt.plot(df['Prediction'], color='r')
    plt.plot(df['Actual'], color='g')
    plt.title(title)
    plt.legend(handles=[mpatches.Patch(color='g', label='Actual'), mpatches.Patch(color='r', label='Prediction')])
    plt.ylabel('Wind Speed(m/s)')
    plt.xlabel('Hours Ahead')
    plt.show()

    md_mape = mape(y[start:end], final_predictions) * 100
    md_rmse = mse(y[start:end], final_predictions, squared=True)
    md_mse = mse(y[start:end], final_predictions, squared=False)

    return md_mape, md_rmse, md_mse


def plot_predictions(model_lstm, model_cnn, model_gru, X, y, title='#####', start=0, end=100):
    try:

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
