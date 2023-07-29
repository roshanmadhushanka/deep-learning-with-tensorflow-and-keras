import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras as k
from keras.layers import Dense, Normalization
import seaborn as sb
import os


def load_data():
    _column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
    _data = pd.read_csv(os.path.join('..', 'datasets', 'auto-mpg.csv'), na_values='?')
    _data = _data.drop(['origin', 'car name'], axis=1)
    _data = _data.dropna()

    _train = _data.sample(frac=0.8, random_state=0)
    _test = _data.drop(_train.index)

    return _train, _test


if __name__ == '__main__':
    train, test = load_data()

    x_train = train.copy()
    x_test = test.copy()

    y_train = x_train.pop('mpg')
    y_test = x_test.pop('mpg')

    # Normalize
    data_normalizer = Normalization(axis=1)
    data_normalizer.adapt(np.array(x_train))

    # Model
    model = k.Sequential([
        data_normalizer,
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation=None)
    ])
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(x=x_train, y=y_train, epochs=100, verbose=1, validation_split=0.2)

    # plt.plot(history.history['loss'], label='loss')
    # plt.plot(history.history['val_loss'],  label='val_loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Error (MPG)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Prediction
    y_pred = model.predict(x_test).flatten()

    # a = plt.axes(aspect='equal')
    # plt.scatter(y_test, y_pred)
    # plt.xlabel('True Values (MPG)')
    # plt.ylabel('Predictions (MPG)')
    # lims = [0, 50]
    # plt.xlim(lims)
    # plt.ylim(lims)
    # plt.plot(lims,  lims)
    # plt.show()

    error = y_pred - y_test
    plt.hist(error, bins=30)
    plt.xlabel('Prediction Error (MPG)')
    plt.ylabel('Count')
    plt.show()


