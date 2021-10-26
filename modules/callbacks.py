import numpy as np
from tensorflow.keras.callbacks import Callback

from boids.evaluate_boids import evaluate_complexity
from boids.forward import forward


class ComplexityCallback(Callback):
    def __init__(
        self, test_every=10, n_trajectories=1, trajectory_len=1000, n_boids=100
    ):
        super().__init__()
        self.test_every = test_every
        self.n_trajectories = n_trajectories
        self.trajectory_len = trajectory_len
        self.n_boids = n_boids
        self.complexities = []

    def on_epoch_begin(self, epoch, logs=None):
        if self.test_every > 0 and epoch == 0:
            self.evaluate_complexity()

    def on_epoch_end(self, epoch, logs=None):
        if self.test_every > 0 and epoch > 0 and epoch % self.test_every == 0:
            self.evaluate_complexity()

    def on_train_end(self, logs=None):
        if self.test_every > 0:
            self.complexities = np.array(self.complexities)
            np.savez("complexities.npz", complexities=self.complexities)

    def evaluate_complexity(self):
        out = evaluate_complexity(
            self.model,
            forward,
            self.n_trajectories,
            self.trajectory_len,
            self.n_boids,
        )
        self.complexities.append(out)
