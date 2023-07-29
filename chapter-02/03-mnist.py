import keras.losses
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras as k
from keras.layers import Dense, Flatten


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(visible=False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel('Pred {} Conf: {:2.0f}% True ({})'.format(predicted_label, 100.0 * np.max(predictions_array), true_label),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(visible=False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


if __name__ == '__main__':
    # Load dataset
    ((x_train, y_train), (x_test, y_test)) = tf.keras.datasets.mnist.load_data()

    # Preprocess
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int32)

    x_test = x_test / np.float32(255)
    y_test = y_test.astype(np.int32)

    # Model
    # model = k.Sequential([
    #     Flatten(input_shape=(28, 28)),
    #     Dense(units=10, activation='sigmoid')
    # ])

    model = k.Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(units=128, activation='relu'),
        Dense(units=10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    # Train
    history = model.fit(x=x_train, y=y_train, epochs=50, verbose=1, validation_split=0.2)

    # Plot
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(visible=True)
    plt.show()

    predictions = model.predict(x=x_test)
    i = 56
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], y_test, x_test)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], y_test)
    plt.show()
