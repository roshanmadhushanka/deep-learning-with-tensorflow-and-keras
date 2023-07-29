import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import keras as k
from keras.layers import Dense


def generate_load():
    np.random.seed(0)
    _area = 2.5 * np.random.randn(100) + 25
    _price = 25 * _area + np.random.randint(20, 50, size=len(_area))
    _data = np.array([_area, _price])
    _data = pd.DataFrame(data=_data.T, columns=['area', 'price'])
    _data = (_data - _data.min()) / (_data.max() - _data.min())
    return _data


if __name__ == '__main__':
    data = generate_load()

    model = k.Sequential()
    model.add(Dense(units=1, input_shape=[1, ], activation=None))
    model.summary()

    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(x=data['area'], y=data['price'], epochs=100, batch_size=32, verbose=1, validation_split=0.2)

    y_pred = model.predict(x=data['area'])
    plt.plot(data['area'], y_pred, color='red', label='Predicted Price')
    plt.scatter(data['area'], data['price'], label='Training Data')
    plt.xlabel('Area')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(os.path.join('data', 'house_price_prediction.png'))
