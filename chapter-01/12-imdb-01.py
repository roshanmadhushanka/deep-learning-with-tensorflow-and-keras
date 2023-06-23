import tensorflow as tf
from keras import datasets, layers, models
from keras.optimizers import Adam
from keras_preprocessing import sequence
import tensorflow_datasets as tfds

MAX_LEN = 200
N_WORDS = 10000
DIM_EMBEDDING = 256
EPOCHS = 20
BATCH_SIZE = 500


def load_data():
    (x_train,  y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=N_WORDS)
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
    return (x_train, y_train),  (x_test, y_test)


def build_model():
    model = models.Sequential()

    # Input : Embedding Layer
    # The model will take as input an integer matrix of size (batch, input_length)
    # The model will output dimension  (input_length, dim_embedding)
    # The largest integer in the input should be no larger than N_WORDS (vocabulary size)
    model.add(layers.Embedding(input_dim=N_WORDS, output_dim=DIM_EMBEDDING, input_length=MAX_LEN))
    model.add(layers.Dropout(rate=0.3))

    # Take the maximum  value of either feature vector from  each of the N_WORDS features
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(units=1, activation='sigmoid'))
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test,  y_test) = load_data()
    model = build_model()
    model.summary()

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    score = model.fit(x=x_train, y=y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))
    score = model.evaluate(x=x_test, y=y_test, batch_size=BATCH_SIZE)
    print(score)
