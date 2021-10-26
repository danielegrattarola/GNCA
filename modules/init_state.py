import numpy as np


class State:
    def __init__(self, n_nodes, channels):
        self.n_nodes = n_nodes
        self.channels = channels

    def __call__(self):
        raise NotImplementedError

    def __repr__(self):
        attrs = ", ".join(
            [
                "{}={}".format(k, getattr(self, k))
                for k in self.__dict__.keys()
                if not k.startswith("__")
            ]
        )
        return "{}({})".format(self.__class__.__name__, attrs)


class SphericalizeState(State):
    def __init__(self, state):
        super().__init__(*state.shape)
        self.state = state / np.linalg.norm(state, axis=-1, keepdims=True)

    def __call__(self):
        return self.state


class Ones(State):
    def __init__(self, n, channels):
        super().__init__(n, channels)
        self.state = np.ones((n, channels))

    def __call__(self):
        return self.state


class Zeros(State):
    def __init__(self, n, channels):
        super().__init__(n, channels)
        self.state = np.zeros((n, channels))

    def __call__(self):
        return self.state


class MiddleOne(State):
    def __init__(self, n, channels):
        super().__init__(n, channels)
        self.state = np.zeros((n, channels))
        self.state[n // 2, :] = 1.0

    def __call__(self):
        return self.state
