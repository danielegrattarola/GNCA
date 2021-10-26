"""
This script trains a MLP with 2 hidden neuron and 1 output neuron to approximate the
transition rule of the Voronoi GCA.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

# Create dataset of transitions
data = []
for mu in np.linspace(0.01, 0.99, 99):
    for x in [0, 1]:
        data.append((x, mu))
data = np.array(data)

thresh = 0.42
y = []
for d in data:
    if d[1] <= thresh:
        y.append(d[0])
    else:
        y.append(1 - d[0])
y = np.array(y)[:, None]


def init(shape, dtype=None):
    """Function to initialize the weights of the MLP"""
    return tf.random.uniform(shape, -1, 1, dtype=dtype)


while True:  # Loop until one MLP converges in at most 5000 epochs
    m = Sequential(
        [
            Dense(
                2,
                "relu",
                kernel_initializer=init,
                bias_initializer=init,
                kernel_regularizer=l2(0.001),
                bias_regularizer=l2(0.001),
            ),
            Dense(
                1,
                "sigmoid",
                kernel_initializer=init,
                bias_initializer=init,
                kernel_regularizer=l2(0.001),
                bias_regularizer=l2(0.001),
            ),
        ]
    )
    m.compile("adam", "mse", metrics=["acc", "mse"])
    h = m.fit(data, y, epochs=5000, batch_size=198, verbose=0)

    if h.history["acc"][-1] < 0.99:
        print(f'Fail {h.history["acc"][-1]}')
    else:
        print("Converged")
        break

# Once the MLP converges, train it even more to get to the lowest possible loss
h = m.fit(
    data,
    y,
    epochs=100000,
    batch_size=198,
    verbose=1,
    callbacks=[
        ReduceLROnPlateau(patience=10000, min_delta=1e-8, verbose=1, monitor="loss")
    ],
)

# Plot the true function and the approximation learned by the MLP
pred = m.predict(data)
plt.figure()
plt.subplot(121)
plt.scatter(*data[:, :2].T, c=y)
plt.subplot(122)
plt.scatter(*data[:, :2].T, c=pred)

# Plot of the true transition included in the paper
plt.figure(figsize=(2, 2))
c = np.array(["gray", "orange"])[y.astype(int)][:, 0]
plt.scatter(*data[:, :2].T, c=c, marker="s")
plt.xticks([0, 1])
plt.xlim(-0.1, 1.1)
plt.xlabel("Current state")
plt.ylabel("Density")
plt.tight_layout()

# Test the set of weights reported in the paper
w = [
    np.array([[-1.98, 1.64], [2.63, -2.8]], dtype=np.float32),
    np.array([-0.46, 0.17], dtype=np.float32),
    np.array([[3.3], [3.3]], dtype=np.float32),
    np.array([-2.1], dtype=np.float32),
]
m.set_weights(w)
m.evaluate(data, y)
