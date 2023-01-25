import datetime

from model_train.train import read_wind_meteodata, df_to_X_y
from model_train.train import train_models
from model_train.predict import four_hour_prediction, plot_predictions
from tensorflow.python.keras.models import load_model


WINDOW_SIZE = 5
start = datetime.datetime.now()

wind = read_wind_meteodata()
X1, y1 = df_to_X_y(wind, WINDOW_SIZE)

train_len = int(len(X1) * 0.7)
test_len = int(len(X1) * 0.9)
X_test1, y_test1 = X1[train_len:test_len], y1[train_len:test_len]


train_models()


model_lsmt = load_model('C:/Users/Maksym_Mysak/PycharmProjects/wind-prediction/prediction_model/lstm/LSTM_model_1_L_1_S')
model_cnn = load_model('C:/Users/Maksym_Mysak/PycharmProjects/wind-prediction/prediction_model/cnn/CNN_model_1_L_1_S')
model_gru = load_model('C:/Users/Maksym_Mysak/PycharmProjects/wind-prediction/prediction_model/gru/GRU_model_1_L_1_S')


print("Test 1")
print(plot_predictions(model_lsmt, model_cnn, model_gru, X_test1, y_test1, 'Test 1', 1340, 1360))
print("Test 2")
print(plot_predictions(model_lsmt, model_cnn, model_gru, X_test1, y_test1, 'Test 2', 1380, 1400))
print("Test 3")
print(plot_predictions(model_lsmt, model_cnn, model_gru, X_test1, y_test1, 'Test 3', 1666, 1670))
print("Test 4")
print(plot_predictions(model_lsmt, model_cnn, model_gru, X_test1, y_test1, 'Test 4', 1810, 1815))
print("Test 5")
print(plot_predictions(model_lsmt, model_cnn, model_gru, X_test1, y_test1, 'Test 5', 1840, 1860))
print("Test 6")
print(plot_predictions(model_lsmt, model_cnn, model_gru, X_test1, y_test1, 'Test 6', 3020, 3060))
print("Test 7")
print(plot_predictions(model_lsmt, model_cnn, model_gru, X_test1, y_test1, 'Test 7', 4240, 4270))
print("Test 8")
print(plot_predictions(model_lsmt, model_cnn, model_gru, X_test1, y_test1, 'Test 8', 4340, 4370))
print("Test 9")
print(plot_predictions(model_lsmt, model_cnn, model_gru, X_test1, y_test1, 'Test 9', 4355, 4375))
print("Test 10")
print(plot_predictions(model_lsmt, model_cnn, model_gru, X_test1, y_test1, 'Test 10', 4235, 4355))
print("Test 11")
print(plot_predictions(model_lsmt, model_cnn, model_gru, X_test1, y_test1, 'Test 11', 60, 64))
print("Test 12")
print(plot_predictions(model_lsmt, model_cnn, model_gru, X_test1, y_test1, 'Test 12', 90, 94))
print("Test 13")
print(plot_predictions(model_lsmt, model_cnn, model_gru, X_test1, y_test1, 'Test 13', 1100, 1104))
print("Test 14")
print(plot_predictions(model_lsmt, model_cnn, model_gru, X_test1, y_test1, 'Test 14', 1130, 1134))
print("Test 15")
print(plot_predictions(model_lsmt, model_cnn, model_gru, X_test1, y_test1, 'To comp 1', 1340, 1344))
print("Test 16")
print(plot_predictions(model_lsmt, model_cnn, model_gru, X_test1, y_test1, 'To comp 2', 1376, 1380))

end = datetime.datetime.now()


print(start)
print(end)