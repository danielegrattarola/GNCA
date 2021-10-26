"""
Trains the GNCA to converge to a target point cloud.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy.ndimage import gaussian_filter1d
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.transforms import NormalizeSphere
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from models import GNNCASimple
from modules.graphs import get_cloud
from modules.init_state import SphericalizeState
from modules.state_cache import StateCache

# tf.config.run_functions_eagerly(True)
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def run(graph):
    @tf.function(experimental_relax_shapes=True)
    def train_step(x, steps):
        print(f"Tracing for {steps} steps")
        with tf.GradientTape() as tape:
            out = model.steps([x, a], steps=steps, training=True)
            loss = loss_fn(y, out) + sum(model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        if args.grad_clip:
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return out, loss

    # Prepare inputs for training
    y = graph.x
    a = sp_matrix_to_sp_tensor(graph.a)
    state_cache = StateCache(
        SphericalizeState(y),
        size=args.cache_size,
        reset_every=args.cache_reset_every,
    )

    # Train
    history = {
        "loss": [],
        "best_loss": np.inf,
        "best_model": None,
        "steps_avg": [],
        "steps_std": [],
        "steps_max": [],
        "steps_min": [],
    }
    best_loss = np.inf
    current_es_patience = args.es_patience
    current_lr_patience = args.lr_patience
    for i in range(args.epochs):
        loss = 0
        for _ in range(args.batches_in_epoch):
            x, idxs = state_cache.sample(args.batch_size)
            x = x.astype("f4")
            steps = np.random.randint(args.min_steps, args.max_steps)

            out, loss_step = train_step(x, tf.constant(steps))
            out = out.numpy()
            loss += loss_step
            # Update cache
            state_cache.update(idxs, out, steps)
            history["steps_avg"].append(np.mean(state_cache.counter))
            history["steps_std"].append(np.std(state_cache.counter))
            history["steps_max"].append(np.max(state_cache.counter))
            history["steps_min"].append(np.min(state_cache.counter))

        loss /= args.batches_in_epoch
        history["loss"].append(loss)

        print(
            f"Iter {i} - Steps: {steps:3d} - Loss: {loss:.10e} - "
            f"ES pat. {current_es_patience} - LR pat {current_lr_patience}"
        )
        if loss + args.tol < best_loss:
            best_loss = loss
            best_model = model.get_weights()
            current_es_patience = args.es_patience
            current_lr_patience = args.lr_patience
            print(f"Loss improved ({best_loss})")
        else:
            current_es_patience -= 1
            current_lr_patience -= 1
            if current_es_patience == 0:
                print("Early stopping")
                model.set_weights(best_model)
                break
            if current_lr_patience == 0:
                print(f"Reducing LR to {optimizer.lr * args.lr_red_factor}")
                current_lr_patience = args.lr_patience
                K.set_value(optimizer.lr, optimizer.lr * args.lr_red_factor)
                model.set_weights(best_model)

    return history, state_cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-3, type=float, help="Initial LR")
    parser.add_argument(
        "--batch_size", default=8, type=int, help="Size of the mini-batches"
    )
    parser.add_argument("--epochs", default=100000, type=int, help="Training epochs")
    parser.add_argument(
        "--batches_in_epoch", default=10, type=int, help="Batches in an epoch"
    )
    parser.add_argument(
        "--es_patience", default=1000, type=int, help="Patience for early stopping"
    )
    parser.add_argument(
        "--lr_patience", default=750, type=int, help="Patience for LR annealing"
    )
    parser.add_argument(
        "--tol", default=1e-6, type=float, help="Tolerance for improvements"
    )
    parser.add_argument(
        "--lr_red_factor", default=0.1, type=float, help="Rate for LR annealing"
    )
    parser.add_argument("--min_steps", default=10, type=int, help="Minimum n. of steps")
    parser.add_argument("--max_steps", default=11, type=int, help="Maximum n. of steps")
    parser.add_argument(
        "--activation", default="tanh", type=str, help="Activation for the GNCA"
    )
    parser.add_argument("--grad_clip", action="store_true", help="Clip the gradient")
    parser.add_argument(
        "--cache_size", default=1024, type=int, help="Size of the cache"
    )
    parser.add_argument(
        "--cache_reset_every",
        default=32,
        type=int,
        help="How often to reset one state in cache",
    )
    parser.add_argument(
        "--outdir", default="results", type=str, help="Where to save output"
    )
    args = parser.parse_args()

    graphs = [
        get_cloud("Grid2d", N1=20, N2=20),
        get_cloud("Bunny"),
        get_cloud("Minnesota"),
        get_cloud("Logo"),
        get_cloud("SwissRoll", N=200),
    ]

    for graph in graphs:
        graph = NormalizeSphere()(graph)

        model = GNNCASimple(activation=args.activation, batch_norm=False)
        optimizer = Adam(learning_rate=args.lr)
        loss_fn = MeanSquaredError()

        history, state_cache = run(graph)

        # Unpack data
        y = graph.x
        a = sp_matrix_to_sp_tensor(graph.a)

        # Run model for the twice the maximum number of steps in the cache
        x = state_cache.initial_state()
        x = x[None, ...]
        steps = 2 * int(np.max(state_cache.counter))
        zs = [x]
        for _ in range(steps):
            z = model([zs[-1], a], training=False)
            zs.append(z.numpy())
        zs = np.vstack(zs)
        z = zs[-1]

        out_dir = f"{args.outdir}/{graph.name}"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/config.txt", "w") as f:
            f.writelines([f"{k}={v}\n" for k, v, in vars(args).items()])
        np.savez(f"{out_dir}/run_point_cloud.npz", y=y, z=z, history=history, zs=zs)

        # Plot difference between target and output points
        plt.figure(figsize=(2.5, 2.5))
        cmap = plt.get_cmap("Set2")
        plt.scatter(*y[:, :2].T, color=cmap(0), marker=".", s=1)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/target.pdf")

        plt.figure(figsize=(2.5, 2.5))
        cmap = plt.get_cmap("Set2")
        plt.scatter(*z[:, :2].T, color=cmap(1), marker=".", s=1)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/endstate.pdf")

        # Plot loss and loss trend
        plt.figure(figsize=(2.6, 2.5))
        cmap = plt.get_cmap("Set2")
        plt.plot(history["loss"], alpha=0.3, color=cmap(0), label="Real")
        plt.plot(gaussian_filter1d(history["loss"], 50), color=cmap(0), label="Trend")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/loss.pdf")

        # Plot change between consecutive state
        plt.figure(figsize=(2.5, 2.5))
        cmap = plt.get_cmap("Set2")
        change = np.abs(zs[:-1] - zs[1:]).mean((1, 2))
        loss = np.array([loss_fn(y, zs[i]).numpy() for i in range(len(zs))])
        plt.plot(change, label="Abs. change", color=cmap(0))
        plt.plot(loss, label="Loss", color=cmap(1))
        plt.xlabel("Step")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/change.pdf")

        # Plot evolution of states
        n_states = 10
        plt.figure(figsize=(n_states * 2.0, 2.1))
        for i in range(n_states):
            plt.subplot(1, n_states, i + 1)
            plt.scatter(*zs[i, :, :2].T, color=cmap(1), marker=".", s=1)
            plt.title(f"t={i}")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/evolution.pdf")

        # Plot the average number of steps for the states in the cache
        plt.figure(figsize=(2.5, 2.5))
        cmap = plt.get_cmap("Set2")
        s_avg, s_std = np.array(history["steps_avg"]), np.array(history["steps_std"])
        s_max, s_min = np.array(history["steps_max"]), np.array(history["steps_min"])
        plt.plot(s_avg, label="Avg.", color=cmap(0))
        plt.fill_between(
            np.arange(len(s_std)),
            s_avg - s_std,
            s_avg + s_std,
            alpha=0.5,
            color=cmap(0),
        )
        plt.plot(s_max, linewidth=0.5, linestyle="--", color="k", label="Max")
        plt.xlabel("Epoch")
        plt.ylabel("Number of steps in cache")
        plt.legend()
        plt.xscale("log")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/steps_in_cache.pdf")

    plt.show()
