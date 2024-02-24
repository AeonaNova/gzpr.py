import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

"""
relevant version, that works
"""

TF_ENABLE_ONEDNN_OPTS=0
# Загрузка данных из текстового файла
data = pd.read_csv('gp.csv', sep=';')

# Transform date and time into digitalized view
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = pd.to_numeric(data['Date'])

# Data normalization
scaler = MinMaxScaler()
data['Price'] = scaler.fit_transform(data['Price'].values.reshape(-1,1))

# Creation characteristics and target variable
X = np.array(data['Price']).reshape(-1, 1)
y = np.roll(X, -40)  # Creation target variable to 2 days ahead

# Creation and training of a neural network
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(1,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=100, batch_size=32)

# Prediction stock price
predictions = model.predict(X)

# Assessing model accuracy
score = model.evaluate(X, y)
print('Loss:', score)

# Predictions and source data view
plt.figure(figsize=(12, 6))
plt.plot(X, predictions, label='Предсказания', marker='x')
plt.xlabel('День')
plt.ylabel('Цена акций')
plt.legend()
plt.show()
