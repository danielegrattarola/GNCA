"""
This scripts aggregates the results from multiple runs to plot the average loss in
training and validation.
Use the --paths flag to pass a list of folders containing the history.pkl file produced
by the script run_boids.py (in the paper we run the experiment 5 times)
"""

import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--paths", nargs="+", type=str, default=["results_1/", "results_2/"]
)
args = parser.parse_args()

print(args.paths)

losses = []
val_losses = []
for p in args.paths:
    h = joblib.load(p + "/history.pkl")
    losses.append(h["loss"])
    val_losses.append(h["val_loss"])

min_len = np.min([len(l) for l in losses])
losses = [l[:min_len] for l in losses]
val_losses = [l[:min_len] for l in val_losses]

losses = np.array(losses)
val_losses = np.array(val_losses)
x = np.arange(losses.shape[-1])

loss_avg = np.mean(losses, 0)
loss_std = np.std(losses, 0)
val_loss_avg = np.mean(val_losses, 0)
val_loss_std = np.std(val_losses, 0)

plt.figure(figsize=(3, 2.7))
cmap = plt.get_cmap("Set2")
plt.plot(loss_avg, label="Train", c=cmap(0))
plt.fill_between(
    x, loss_avg - loss_std, loss_avg + loss_std, color=cmap(0), alpha=0.5, linewidth=0
)
plt.plot(val_loss_avg, label="Valid", c=cmap(1))
plt.fill_between(
    x,
    val_loss_avg - val_loss_std,
    val_loss_avg + val_loss_std,
    color=cmap(1),
    alpha=0.5,
    linewidth=0,
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.yscale("log")
plt.tight_layout()
plt.savefig("run_boids_loss_v_epoch.pdf", bbox_inches="tight")
