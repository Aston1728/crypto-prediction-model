#importing the libraries
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from binance import Client

client = Client('api_key','api_secret_key')

#naming the columns and changing them to float
def getminutedata(symbol, interval, lookback):
    frame = pd.DataFrame(client.get_historical_klines(symbol, interval, lookback+' min ago UTC'))
    frame = frame.iloc[:,:6]
    frame.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    frame = frame.set_index('Time')
    frame.index = pd.to_datetime(frame.index, unit='ms')
    frame = frame.astype(float)
    return frame
    
data_final = getminutedata('BTCUSDT','5m', '21600')
    
scalar = MinMaxScaler(feature_range=(0,1))
scaled_data = scalar.fit_transform(data_final['Close'].values.reshape(-1,1))

time_interval_to_train = 24
prediction_interval = 12

x_train = []
y_train = []

for i in range(time_interval_to_train, len(scaled_data) - prediction_interval):
    x_train.append(scaled_data[i - time_interval_to_train: i, 0])
    y_train.append(scaled_data[i + prediction_interval, 0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#LSTM model is applied
model = Sequential()

model.add(LSTM(128, return_sequences= True, input_shape = (x_train.shape[1], 1), activation = 'relu'))
model.add(Dropout(0.4))
model.add(LSTM(64, return_sequences= True, activation = 'relu'))
model.add(Dropout(0.3))
model.add(LSTM(32, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 10, batch_size = 64)

bitcoin_prices = data_final['Close'].values

test_inputs = data_final['Close'].values
test_inputs = test_inputs.reshape(-1,1)
model_inputs = scalar.fit_transform(test_inputs)

x_test = []

for x in range(time_interval_to_train, len(model_inputs)):
    x_test.append(model_inputs[x - time_interval_to_train:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scalar.inverse_transform(prediction_prices)

prediction_prices = np.insert(prediction_prices, 0, np.repeat(np.nan,time_interval_to_train))
