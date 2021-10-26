"""
This script creates a table with the lowest error achieved by the GNCA on the
autonomously-evolved trajectories saved in the output folders.
"""
import argparse
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--paths", nargs="+", default=["results"])
parser.add_argument(
    "--datasets",
    nargs="+",
    default=["Grid2d", "Bunny", "Minnesota", "Logo", "SwissRoll"],
)
args = parser.parse_args()

loss_fn = MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

df = {d: [] for d in args.datasets}
for p in args.paths:
    for d in tqdm(args.datasets):
        f = glob.glob(f"{p}/{d}/run_point_cloud.npz")
        if len(f) == 0:
            df[d].append(None)
            continue
        else:
            f = f[0]
        data = np.load(f, allow_pickle=True)
        y, z, zs, history = data["y"], data["z"], data["zs"], data["history"]
        loss = loss_fn(y[None, ...], zs).numpy().mean(-1)
        df[d].append(loss.min())

df = pd.DataFrame(df, index=args.paths)


def formatter(s):
    try:
        out = f"{s:.2e}"
        exp = int(out[-3:])
        out = out[:-4] + fr"$\cdot 10^{{{exp}}}$"
    except:
        return None

    return out


df = df.applymap(formatter)

with pd.option_context("max_colwidth", 1000):
    print(df.to_latex(escape=False, bold_rows=True))
df.to_csv("lowest_error.csv")
