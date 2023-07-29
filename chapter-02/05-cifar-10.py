import tensorflow as tf
from tensorflow import keras
from keras import optimizers, models, layers, utils

IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32
BATCH_SIZE = 128
EPOCHS = 20
CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIMIZER = optimizers.RMSprop()


def build(input_shape, classes):
    """
    Build model
    :param input_shape: Input shape
    :param classes: Number of classes
    :return: Built model
    """

    model = models.Sequential()
    model.add(layers.Convolution2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10,
                           activation='softmax'))  # used as an activation function  for multi class classification
    # problems
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = utils.to_categorical(y_train, CLASSES)
    print(x_train.shape)
    print(y_train.shape)
    model = build((IMG_ROWS, IMG_COLS, IMG_CHANNELS), CLASSES)
    model.summary()

    # Train
    model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
    model.fit(x=x_train, y=y_train,  batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT,
              verbose=VERBOSE)

    score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
    print('Test Score    :  {}'.format(score[0]))
    print('Test Accuracy :  {}'.format(score[1]))
