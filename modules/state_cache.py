import numpy as np

from modules.utils import is_iterable


class StateCache:
    def __init__(self, initial_state, size=1024, reset_every=32):
        self.initial_state = initial_state
        self.reset_every = reset_every

        self.cache = np.array([initial_state() for _ in range(size)])
        self.counter = np.array([1] * size)

    def sample(self, amount):
        idxs = np.random.randint(len(self.cache), size=amount)
        if amount == 1:
            idxs = int(idxs[0])
        return self.__getitem__(idxs), idxs

    def update(self, idxs, states, counts):
        # Replace one state at random with the initial one
        idx = np.random.randint(0, self.reset_every)
        if idx < len(states):
            states[idx] = self.initial_state()

        # Update state cache
        self.cache[idxs] = states

        # Update counter
        self.count(idxs, counts)
        if idx < len(states):
            self.reset_counter(idxs[idx])

    def count(self, idxs, steps):
        """
        Increases the counter at the given indices by `step`.
        """
        self.counter[idxs] += steps

    def reset_counter(self, idxs):
        """
        Resets the counter at the given indices to 1.
        """
        self.counter[idxs] = 1

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.cache[item]
        elif is_iterable(item):
            return np.array([self.cache[i] for i in item])
        else:
            raise ValueError(f"Unsupported key type: {type(item)}")

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.cache[key] = value
        elif is_iterable(key):
            assert len(value) == len(key), "Indices must have same length as values"
            for i, j in enumerate(key):
                self.cache[j] = value[i]
        else:
            raise ValueError(f"Unsupported key type: {type(key)}")

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)
