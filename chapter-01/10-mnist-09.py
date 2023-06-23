# Find best batch size
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Training parameters
EPOCHS = 20
BATCH_SIZE = 1024
VERBOSE = 1
NB_CLASSES = 10  # Number of outputs
N_HIDDEN = 1024
VALIDATION_SPLIT = 0.2  # How much train data preserve for validation
RESHAPED = 784
DROPOUT = 0.1
LEARNING_RATE = 0.001


def train():
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
    model.add(tf.keras.layers.Dense(N_HIDDEN, input_shape=(RESHAPED,), name='dense_layer_1', activation='relu'))
    model.add(tf.keras.layers.Dropout(DROPOUT))
    model.add(tf.keras.layers.Dense(N_HIDDEN, name='dense_layer_2', activation='relu'))
    model.add(tf.keras.layers.Dense(NB_CLASSES, name='dense_layer_3', activation='softmax'))

    # Summary of the model
    model.summary()

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Compiling the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE,
              validation_split=VALIDATION_SPLIT)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test)
    return test_acc


if __name__ == '__main__':
    max_accuracy_batch_size = 0
    max_accuracy = 0
    for BATCH_SIZE in [64, 128, 256, 512, 1024]:
        accuracy = train()
        print('Hidden Layer : {} -> Accuracy : {}'.format(N_HIDDEN, accuracy))
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_accuracy_batch_size = BATCH_SIZE
    print('Max Accuracy               : {}'.format(max_accuracy))
    print('Batch Size at Max Accuracy : {}'.format(max_accuracy_batch_size))