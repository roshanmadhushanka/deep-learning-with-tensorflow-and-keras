from tensorflow import keras
from keras import layers, models, optimizers
from tensorflow import keras

EPOCHS = 5
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = optimizers.Adam()
VALIDATION_SPLIT = 0.2
IMG_ROWS, IMG_COLS = 28, 28
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
NB_CLASSES = 10


def build(input_shape, classes):
    model = models.Sequential()
    model.add(layers.Conv2D(20, (5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(50, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))
    return model


if __name__ == '__main__':
    # shuffle data and split into train and test
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # reshape
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))

    # cast
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, NB_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NB_CLASSES)

    # initialize optimizer and model
    model = build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        keras.callbacks.TensorBoard(log_dir='logs')
    ]

    # fit
    # history = model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE,
    #                     validation_split=VALIDATION_SPLIT, callbacks=callbacks)
    history = model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE,
                        validation_split=VALIDATION_SPLIT)
    score = model.evaluate(x=x_test, y=y_test, verbose=VERBOSE)
    print('Test Score : {}'.format(score[0]))
    print('Test Accuracy : {}'.format(score[1]))

    