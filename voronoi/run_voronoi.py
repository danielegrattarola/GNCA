import tensorflow as tf
from scipy import stats
from spektral.layers.ops import sp_matrix_to_sp_tensor
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from models import GNNCASimple
from modules.ca import *

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

    # Prepare inputs for training
    a = sp_matrix_to_sp_tensor(gca.graph.a)

    # Train
    history = []
    for epoch in tqdm(range(epochs)):
        state, next_state = get_batch(batch_size)
        out, loss, acc = train_step(state, next_state)

        # Test out-of-sample validation
        state_val, next_state_val = get_batch(batch_size)
        out_val, loss_val, acc_val = evaluate(state_val, next_state_val)

        history.append((loss, loss_val, acc, acc_val))
        tqdm.write(
            f"Epoch {epoch} Loss = {loss:.3e}, Acc = {100*acc:.2f}, "
            f"Loss val = {loss_val:.3e}, Acc val={100*acc_val:.2f}  "
        )

    return np.array(history), model


if __name__ == "__main__":
    # Configuration
    n_cells = 1000
    threshold = 0.42
    epochs = 1000
    batch_size = 32
    repetitions = 1

    histories = []
    for _ in range(repetitions):
        gca = VoronoiCA(n_cells, mu=0, sigma=threshold)
        history, model = run(gca)
        histories.append(history)
    np.savez("results/learn_gca_loss_v_epoch.npz", histories=histories)

    hist_mean = np.median(histories, 0)
    hist_std = stats.median_absolute_deviation(histories, 0)

    plt.figure(figsize=(3.1, 3))
    cmap = plt.get_cmap("Set2")
    x = np.arange(hist_mean.shape[0])
    plt.plot(hist_mean[:, 0], label="Train", c=cmap(0))
    plt.fill_between(
        x,
        hist_mean[:, 0] - hist_std[:, 0],
        hist_mean[:, 0] + hist_std[:, 0],
        color=cmap(0),
        alpha=0.5,
        linewidth=0.0,
    )
    plt.plot(hist_mean[:, 1], label="Valid", c=cmap(1))
    plt.fill_between(
        x,
        hist_mean[:, 1] - hist_std[:, 1],
        hist_mean[:, 1] + hist_std[:, 1],
        color=cmap(1),
        alpha=0.5,
        linewidth=0.0,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("results/learn_gca_loss_v_epoch.pdf", bbox_inches="tight")

    plt.figure(figsize=(3, 3))
    cmap = plt.get_cmap("Set2")
    plt.plot(hist_mean[:, 2], label="Train", c=cmap(0))
    plt.fill_between(
        x,
        hist_mean[:, 2] - hist_std[:, 2],
        hist_mean[:, 2] + hist_std[:, 2],
        color=cmap(0),
        alpha=0.5,
        linewidth=0.0,
    )
    plt.plot(hist_mean[:, 3], label="Valid", c=cmap(1))
    plt.fill_between(
        x,
        hist_mean[:, 3] - hist_std[:, 3],
        hist_mean[:, 3] + hist_std[:, 3],
        color=cmap(1),
        alpha=0.5,
        linewidth=0.0,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig("results/learn_gca_acc_v_epoch.pdf", bbox_inches="tight")

    plt.show()
