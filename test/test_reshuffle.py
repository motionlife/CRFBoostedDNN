import tensorflow as tf
from tensorflow import keras
import numpy as np

X = np.arange(7).astype(np.float32).reshape(-1, 1)
y = X ** 2
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(100, reshuffle_each_iteration=True)
# dataset = dataset.repeat(2)
dataset = dataset.batch(2)


@tf.function
def log_inputs(inputs):
    tf.print(inputs)
    return inputs


model = keras.models.Sequential([
    keras.layers.Lambda(log_inputs),
    keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer="sgd")
model.fit(dataset, epochs=5, verbose=0)
