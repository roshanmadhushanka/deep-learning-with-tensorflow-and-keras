import tensorflow as tf
import numpy as np
from tensorflow import keras

# Training parameters
EPOCHS = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10  # Number of outputs
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2  # How much train data preserve for validation
RESHAPED = 784

if __name__ == '__main__':
    # Loading  MNIST dataset
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # x_train is 60,000 of 28 x 28 values -> reshape it ti  60,000  X  784
    # x_train.shape = (60000, 28, 28) -> (60000, 784)
    x_train = x_train.reshape(60000, RESHAPED)
    x_train = x_train.astype('float32')
    x_test = x_test.reshape(10000, RESHAPED)
    x_test = x_test.astype('float32')

    # Normalize inputs to bee within
    x_train /= 255
    x_test /= 255
    print('Train samples  : {}'.format(x_train.shape[0]))
    print('Test samples   : {}'.format(x_test.shape[0]))

    # One hot representation of the labels
    y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)

    # Build the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(NB_CLASSES, input_shape=(RESHAPED, ), name='dense_layer', activation='softmax'))

    # Compiling the model
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE,
              validation_split=VALIDATION_SPLIT)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test)
    print('Accuracy : {}'.format(test_acc))
    print('Loss     : {}'.format(test_loss))
    