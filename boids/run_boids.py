"""
Trains the GNCA to imitate the Boids GCA.
"""
import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from spektral.data import DisjointLoader
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from boids.evaluate_boids import evaluate
from boids.forward import forward
from models.gnn_ca_simple_boids import GNNCASimpleBoids
from modules.boids import make_dataset
from modules.callbacks import ComplexityCallback

# tf.config.run_functions_eagerly(True)
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def run(data_tr, data_va, data_te):
    model = GNNCASimpleBoids(
        activation="linear",
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="mean",
    )
    optimizer = Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss="mse")

    loader_tr = DisjointLoader(data_tr, node_level=True, batch_size=args.batch_size)
    loader_va = DisjointLoader(data_va, node_level=True, batch_size=args.batch_size)
    loader_te = DisjointLoader(data_te, node_level=True, batch_size=args.batch_size)

    history = model.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        epochs=1000000,
        validation_data=loader_va.load(),
        validation_steps=loader_va.steps_per_epoch,
        callbacks=[
            EarlyStopping(
                patience=args.es_patience, restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(patience=args.lr_patience, min_delta=1e-8, verbose=1),
            ComplexityCallback(test_every=args.test_complexity_every),
        ],
    )

    results_te = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)

    return history, results_te, model


####################################################################################
# Configuration
####################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=1e-3, type=float, help="Initial LR")
parser.add_argument(
    "--batch_size", default=30, type=int, help="Size of the mini-batches"
)
parser.add_argument(
    "--es_patience", default=20, type=int, help="Patience for early stopping"
)
parser.add_argument(
    "--lr_patience", default=10, type=int, help="Patience for LR annealing"
)
parser.add_argument(
    "--lr_red_factor", default=0.1, type=float, help="Rate for LR annealing"
)
parser.add_argument(
    "--n_boids", default=100, type=int, help="N. of boids in simulation"
)
parser.add_argument(
    "--trajectory_len", default=300, type=int, help="Length of trajectories"
)
parser.add_argument(
    "--tr_set_size", default=300, type=int, help="N. of training trajectories"
)
parser.add_argument(
    "--va_set_size", default=30, type=int, help="N. of valid. trajectories"
)
parser.add_argument(
    "--te_set_size", default=30, type=int, help="N. of test trajectories"
)
parser.add_argument(
    "--test_complexity_every",
    default=-1,
    type=int,
    help="How often to test for complexity (-1 for never)",
)
args = parser.parse_args()
with open("config.txt", "w") as f:
    f.writelines([f"{k}={v}\n" for k, v, in vars(args).items()])

####################################################################################
# Training
####################################################################################
data_tr = make_dataset(
    args.tr_set_size, args.trajectory_len, n_boids=args.n_boids, n_jobs=-1
)
data_va = make_dataset(
    args.va_set_size, args.trajectory_len, n_boids=args.n_boids, n_jobs=-1
)
data_te = make_dataset(
    args.te_set_size, args.trajectory_len, n_boids=args.n_boids, n_jobs=-1
)

history, results_te, model = run(data_tr, data_va, data_te)
print(f"Test loss: {results_te}")
model.save("best_model")
joblib.dump(history.history, "history.pkl")

####################################################################################
# Evaluation
####################################################################################
trajectory_len = 1000
n_boids = 100
init_blob = False
evaluate(model, forward, trajectory_len, n_boids, init_blob=init_blob)

####################################################################################
# Plot SampEn and Correlation Dimension
####################################################################################
if args.test_complexity_every > 0:
    c = np.load("complexities.npz")["complexities"]
    means = c[:, 0, :]
    stds = c[:, 1, :]
    x = np.arange(means.shape[0] - 1) * 10

    plt.figure(figsize=(4.5, 2))
    plt.subplot(121)
    plt.axhline(means[:, 0].mean(), label="True")
    plt.plot(x, means[:-1, 1], " x", label="GNCA")
    plt.xlabel("Epoch")
    plt.ylabel("SampEn")
    plt.xticks(x[::2])
    plt.legend()

    plt.subplot(122)
    plt.axhline(means[:, 2].mean(), label="True")
    plt.plot(x, means[:-1, 3], " x", label="GNCA")
    plt.xlabel("Epoch")
    plt.ylabel("CD")
    plt.xticks(x[::2])
    plt.legend()
    plt.tight_layout()

    plt.savefig("complexities.pdf", bbox_inches="tight")
