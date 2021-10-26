import numpy as np


def entropy(p, axis=0):
    return -np.sum(p * np.log2(p + 1e-9), axis=axis)


def shannon_entropy(states: np.ndarray) -> np.ndarray:
    """
    Computes the entropy of each cell over time.
    :param states: a sequence of state configurations of shape (n_steps, n_cells)
    :return: entropy over time for each cell, of shape (n_cells, )
    """
    unique_states = np.unique(states)
    total_steps = len(states)
    probabilities = []  # (n_states, n_cells)
    for uniq_state in unique_states:
        p = (states == uniq_state).sum(0) / total_steps  # (n_cells, )
        probabilities.append(p)
    probabilities = np.array(probabilities)
    H = -np.sum(probabilities * np.log2(probabilities + 1e-9), axis=0)  # (n_cells, )

    return H


def word_entropy(states: np.ndarray):
    n_steps, n_cells = states.shape
    probabilities = []  # (n_cells, n_words)
    for cell in range(n_cells):
        counts = get_total_word_counts(states[:, cell])
        counts = counts[1:]  # Remove count of 0-lenght words
        total_words = counts.sum()
        p = counts / total_words  # (n_words, )
        probabilities.append(p)

    H = np.array([entropy(p) for p in probabilities])

    return H


def get_word_counts(arr, value):
    isvalue = np.concatenate(([0], np.equal(arr, value).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isvalue))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    count = ranges[:, 1] - ranges[:, 0]
    return np.bincount(count)


def get_total_word_counts(arr):
    counts = []
    unique_states = np.unique(arr)
    for value in unique_states:
        counts.append(get_word_counts(arr, value))

    max_word_lenght = max(len(c) for c in counts)
    counts_padded = np.zeros((unique_states.shape[0], max_word_lenght))
    for i, c in enumerate(counts):
        counts_padded[i, : len(c)] = c

    return counts_padded.sum(0)
