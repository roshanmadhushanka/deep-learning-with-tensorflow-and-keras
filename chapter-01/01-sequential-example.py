import tensorflow as tf
from tensorflow import keras

NB_CLASSES = 10
RESHAPED = 784

model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(NB_CLASSES, input_shape=(RESHAPED,),  kernel_initializer='zeros', name='dense_layer',
                             activation='softmax'))
# each neuron can initialized specific weights via the `kernel_initializer`
## random_uniform
## random_normal
## zero

