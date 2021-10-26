"""
Evaluates the trained GNCA by comparing it to the true Boids GCA.
"""

import matplotlib.pyplot as plt
import nolds
import numpy as np
import tensorflow as tf
from spektral.data import DisjointLoader
from spektral.layers import ops
from tensorflow.keras.models import load_model

from modules.boids import make_dataset


@tf.function(experimental_relax_shapes=True)
def forward(model, x, a, i, training=None):
    """Computes one forward pass of the GNCA"""
    x_pred = model((x, a, i[:, None]), training=training)
    return x_pred


def avg_measure(trajectory, measure_fn, n_boids=None, coord=0, **kwargs):
    n_boids_total = trajectory.shape[-2]
    measures = []
    for i in np.random.permutation(n_boids_total)[:n_boids]:
        measures.append(measure_fn(trajectory[:, i, coord], **kwargs))

    mn, std = np.mean(measures), np.std(measures)
    print(f"{measure_fn.__name__} {mn} +- {std}")
    return np.array(measures)


def evaluate(model, forward, trajectory_len, n_boids, init_blob=False):
    np.random.seed(0)
    init = None
    if init_blob:
        init = [
            0.1 * np.random.randn(n_boids, 2),
            0.001 + 0.005 * np.random.rand(n_boids, 2),
        ]

    data_te, boids_te = make_dataset(
        1,
        trajectory_len,
        return_boids=True,
        n_boids=n_boids,
        n_jobs=-1,
        init=init,
    )
    loader_te = DisjointLoader(data_te, node_level=True, epochs=1, shuffle=False)

    boid_trajectory_true = []
    boid_trajectory_pred = []
    boid_trajectory_auto = []
    avg_degree_trajectory_true = []
    avg_degree_trajectory_auto = []
    for sample in loader_te:
        inputs, x_next = sample
        x_next_pred = forward(model, *inputs, training=False)
        avg_degree_trajectory_true.append(np.average(ops.degrees(inputs[1]).numpy()))
        if len(boid_trajectory_auto) == 0:
            boid_trajectory_auto.append(x_next_pred)
        else:
            x_last = boid_trajectory_auto[-1]
            a = boids_te.get_neighbors(x_last[:, :2])
            a = ops.sp_matrix_to_sp_tensor(a)
            avg_degree_trajectory_auto.append(np.average(ops.degrees(a).numpy()))

            inputs = [x_last, a, inputs[-1]]
            x_next_auto = forward(model, *inputs, training=False)
            boid_trajectory_auto.append(x_next_auto)

        boid_trajectory_true.append(x_next)
        boid_trajectory_pred.append(x_next_pred.numpy())

    boid_trajectory_true = np.array(boid_trajectory_true)
    boid_trajectory_pred = np.array(boid_trajectory_pred)
    boid_trajectory_auto = np.array(boid_trajectory_auto)

    measures = []
    print("True values")
    measures.append(avg_measure(boid_trajectory_true, nolds.sampen))
    measures.append(avg_measure(boid_trajectory_true, nolds.corr_dim, emb_dim=10))
    print("Auto values")
    measures.append(avg_measure(boid_trajectory_auto, nolds.sampen))
    measures.append(avg_measure(boid_trajectory_auto, nolds.corr_dim, emb_dim=10))

    plt.figure()
    for boid_to_track in np.random.permutation(boid_trajectory_auto.shape[-2])[:5]:
        plt.plot(*boid_trajectory_true[:, boid_to_track, :2].T, label="True", c="k")
        plt.plot(*boid_trajectory_pred[:, boid_to_track, :2].T, label="GNCA", c="g")
    plt.legend()
    plt.savefig("boids_pred.pdf")

    plt.figure()
    for boid_to_track in np.random.permutation(boid_trajectory_auto.shape[-2])[:5]:
        plt.plot(*boid_trajectory_true[:, boid_to_track, :2].T, label="True", c="k")
        plt.plot(*boid_trajectory_auto[:, boid_to_track, :2].T, label="GNCA", c="g")
    plt.legend()
    plt.savefig("boids_auto.pdf")

    plt.figure(figsize=(10, 4))
    # Track the best boid
    boid_to_track = (
        ((boid_trajectory_true - boid_trajectory_auto) ** 2).mean(0).mean(-1).argmin()
    )
    # boid_to_track = 0
    ylabels = ["$p_x$", "$p_y$", "$v_x$", "$v_y$"]
    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.plot(boid_trajectory_true[:, boid_to_track, i].T, label="True", c="k")
        plt.plot(boid_trajectory_auto[:, boid_to_track, i].T, label="GNCA", c="g")
        plt.ylabel(ylabels[i])
        if i < 3:
            plt.xticks([])
        else:
            plt.xlabel("Step")
    plt.legend()
    plt.tight_layout()
    plt.savefig("boids_auto_individual_feature.pdf")

    plt.figure()
    plt.plot(avg_degree_trajectory_true, label="True", c="k")
    plt.plot(avg_degree_trajectory_auto, label="GNCA", c="g")
    plt.ylabel("Average degree")
    plt.xlabel("Step")
    plt.legend()
    plt.savefig("boids_avg_degree.pdf")

    plt.show()

    # Show animation of auto-trajectory
    for i in range(len(boid_trajectory_true)):
        boids_te.plot(
            np.concatenate(
                (boid_trajectory_true[i][..., :2], boid_trajectory_auto[i][..., :2]),
                axis=0,
            ),
            c=np.array(
                [0] * len(boid_trajectory_true[i]) + [1] * len(boid_trajectory_auto[i])
            ),
        )
        plt.xticks([])
        plt.yticks([])
        plt.gcf().set_size_inches(2.5, 2.5)
        plt.tight_layout()


def evaluate_complexity(
    model, forward, te_set_size, trajectory_len, n_boids, init_blob=False
):
    np.random.seed(0)
    measures = []
    for i in range(te_set_size):
        init = None
        if init_blob:
            init = [
                0.1 * np.random.randn(n_boids, 2),
                0.001 + 0.005 * np.random.rand(n_boids, 2),
            ]

        data_te, boids_te = make_dataset(
            1,
            trajectory_len,
            return_boids=True,
            n_boids=n_boids,
            n_jobs=-1,
            init=init,
        )
        loader_te = DisjointLoader(data_te, node_level=True, epochs=1, shuffle=False)

        boid_trajectory_true = []
        boid_trajectory_pred = []
        boid_trajectory_auto = []
        avg_degree_trajectory_true = []
        avg_degree_trajectory_auto = []
        for sample in loader_te:
            inputs, x_next = sample
            x_next_pred = forward(model, *inputs, training=False)
            avg_degree_trajectory_true.append(
                np.average(ops.degrees(inputs[1]).numpy())
            )
            if len(boid_trajectory_auto) == 0:
                boid_trajectory_auto.append(x_next_pred)
            else:
                x_last = boid_trajectory_auto[-1]
                a = boids_te.get_neighbors(x_last[:, :2])
                a = ops.sp_matrix_to_sp_tensor(a)
                avg_degree_trajectory_auto.append(np.average(ops.degrees(a).numpy()))

                inputs = [x_last, a, inputs[-1]]
                x_next_auto = forward(model, *inputs, training=False)
                boid_trajectory_auto.append(x_next_auto)

            boid_trajectory_true.append(x_next)
            boid_trajectory_pred.append(x_next_pred.numpy())

        boid_trajectory_true = np.array(boid_trajectory_true)
        boid_trajectory_pred = np.array(boid_trajectory_pred)
        boid_trajectory_auto = np.array(boid_trajectory_auto)

        measures.append(
            (
                avg_measure(boid_trajectory_true, nolds.sampen),
                avg_measure(boid_trajectory_auto, nolds.sampen),
                avg_measure(boid_trajectory_true, nolds.corr_dim, emb_dim=10),
                avg_measure(boid_trajectory_auto, nolds.corr_dim, emb_dim=10),
            )
        )  # (reps, 4, n_boids)

    measures = np.array(measures)
    measures_mean = np.mean(measures, (0, -1))
    measures_std = np.std(measures, (0, -1))
    print(f"SampEn True: {measures_mean[0]} +- {measures_std[0]}")
    print(f"SampEn GNCA: {measures_mean[1]} +- {measures_std[1]}")
    print(f"Corr. dim. True: {measures_mean[2]} +- {measures_std[2]}")
    print(f"Corr. dim. GNCA: {measures_mean[3]} +- {measures_std[3]}")

    return measures_mean, measures_std


if __name__ == "__main__":
    trajectory_len = 1000
    n_boids = 100
    model = load_model("best_model")
    evaluate(model, forward, trajectory_len, n_boids)
