"""
This script can be used to generate all the plots for the fixed target experiment.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tensorflow.keras.losses import MeanSquaredError

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path", default="results/Grid2d/", help="Path to an output folder"
)
parser.add_argument(
    "--nstates",
    type=int,
    default=10,
    help="Number of steps to plot when showing the trajectory",
)
parser.add_argument(
    "--cut",
    action="store_true",
    help="Cut the plot of the error at 50 steps (useful to show details)",
)
parser.add_argument(
    "--show",
    action="store_true",
    help="Show the plots (instead of just saving them to file)",
)
args = parser.parse_args()

out_dir = args.path
cut = 50 if args.cut else None

data = np.load(f"{out_dir}/run_point_cloud.npz", allow_pickle=True)
y, z, zs, history = data["y"], data["z"], data["zs"], data["history"]
history = history.item()
loss_fn = MeanSquaredError()

# Plot target state and final state of trajectory
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
# plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(f"{out_dir}/loss.pdf")

# Plot loss between current state and target
plt.figure(figsize=(2.5, 2.5))
cmap = plt.get_cmap("Set2")
loss = np.array([loss_fn(y, zs[i]).numpy() for i in range(len(zs))])
plt.plot(loss[:cut], label="Error", color=cmap(1))
plt.xlabel("Step")
if not args.cut:
    plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(f"{out_dir}/change.pdf")

# Plot trajectory
n_states = args.nstates
grid_cols = 10
grid_rows = int(np.ceil(n_states / grid_cols))
plt.figure(figsize=(grid_cols * 2.0, grid_rows * 2.0))
for i in range(0, n_states):
    plt.subplot(grid_rows, grid_cols, i + 1)
    plt.scatter(*zs[i + 1, :, :2].T, color="k", marker=".", s=1)
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.title(f"t={i + 1}")
plt.tight_layout()
plt.savefig(f"{out_dir}/evolution.pdf")
plt.savefig(f"{out_dir}/evolution.png")

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
# plt.xscale("log")
plt.tight_layout()
plt.savefig(f"{out_dir}/steps_in_cache.pdf")

if args.show:
    plt.show()
