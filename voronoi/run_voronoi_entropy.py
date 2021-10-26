"""
This script is equivalent to run_voronoi.py but it evaluates the entropy of the current
GNCA after each training step.
"""
import tensorflow as tf
from spektral.layers.ops import sp_matrix_to_sp_tensor
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from models import GNNCASimple
from modules.ca import *
from voronoi.measures import shannon_entropy, word_entropy

# tf.config.run_functions_eagerly(True)
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def run(gca):
    model = GNNCASimple(activation="sigmoid", batch_norm=False)
    optimizer = Adam(learning_rate=1e-2)
    loss_fn = BinaryCrossentropy()

    @tf.function
    def train_step(state, next_state):
        with tf.GradientTape() as tape:
            out = model([state, a], training=True)
            loss = loss_fn(next_state, out) + sum(model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        acc = tf.reduce_mean(binary_accuracy(next_state, out))
        return out, loss, acc

    @tf.function
    def evaluate(state, next_state):
        out = model([state, a], training=False)
        loss = loss_fn(next_state, out)
        acc = tf.reduce_mean(binary_accuracy(next_state, out))

        return out, loss, acc

    def get_batch(batch_size):
        state = np.random.randint(0, 2, (batch_size, n_cells, 1)).astype("f4")
        next_state = np.array([gca.step(s) for s in state])

        return state, next_state

    def evolve_gnnca(state, steps):
        states = [state]
        for _ in range(steps):
            next_state = model([states[-1][None, ...], a], training=False)[0]
            next_state = np.round(next_state.numpy())
            states.append(next_state)

        return np.array(states)

    # Prepare inputs for training
    a = sp_matrix_to_sp_tensor(gca.graph.a)

    init_state_test = np.random.randint(0, 2, (n_cells, 1)).astype("f4")
    states_test_true = gca.evolve(init_state_test, steps=steps)
    Hs_test_true = np.mean(shannon_entropy(states_test_true[..., 0]))
    Hw_test_true = np.mean(word_entropy(states_test_true[..., 0]))

    # Train
    entropies = []
    for epoch in tqdm(range(epochs)):
        state, next_state = get_batch(batch_size)
        out, loss, acc = train_step(state, next_state)

        states_test_pred = evolve_gnnca(init_state_test, steps)
        Hs_test_pred = np.mean(shannon_entropy(states_test_pred[..., 0]))
        Hw_test_pred = np.mean(word_entropy(states_test_pred[..., 0]))

        entropies.append((Hs_test_pred, Hw_test_pred))
        tqdm.write(f"Epoch {epoch} Loss = {loss:.4f}, Acc = {100 * acc:.1f}")

    return np.array((Hs_test_true, Hw_test_true)), np.array(entropies), model


if __name__ == "__main__":
    # Configuration
    n_cells = 1000
    steps = 1000
    threshold = 0.42
    epochs = 100
    batch_size = 32

    gca = VoronoiCA(n_cells, mu=0, sigma=threshold)
    H_true, H_pred, model = run(gca)
    np.savez("results/learn_gca_entropy_change.npz", H_true=H_true, H_pred=H_pred)

    plt.figure(figsize=(3, 3))
    cmap = plt.get_cmap("cool")
    plt.plot(*H_pred.T, color="#333333", linestyle="dashed", linewidth=0.5, alpha=0.5)
    plt.scatter(
        *H_pred.T,
        color=cmap(np.linspace(0, 1, len(H_pred))),
        marker=".",
    )
    plt.scatter(*H_true, color=cmap(1.0), marker="x", label="True", s=40)
    plt.xlabel("$H_s$")
    plt.ylabel("$H_w$")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("results/learn_gca_entropy_change_scatter.pdf", bbox_inches="tight")

    plt.figure(figsize=(3, 3))
    cmap = plt.get_cmap("Set2")
    plt.plot(H_pred[:, 0].T, color=cmap(0), label="$H_s$")
    plt.plot(H_pred[:, 1].T, color=cmap(1), label="$H_w$")
    plt.scatter(len(H_pred), H_true[0], color=cmap(0), marker="x", label="True", s=40)
    plt.scatter(len(H_pred), H_true[1], color=cmap(1), marker="x", s=40)
    plt.xlabel("Epoch")
    plt.ylabel("Entropy")
    plt.legend()
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig("results/learn_gca_entropy_change_plot.pdf", bbox_inches="tight")

    plt.show()
