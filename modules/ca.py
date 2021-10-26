from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from scipy.spatial import Voronoi
from spektral.data import Graph
from spektral.transforms import Delaunay, NormalizeAdj

from modules import voronoi_utils


class CA:
    def __init__(
        self,
        n_cells,
        mu: float = 0.3,
        sigma: float = 0.2,
        n_states: Union[int, str] = 2,
    ):
        self.n_cells = n_cells
        self.mu = mu
        self.sigma = sigma
        self.n_states = n_states
        self.graph = self.get_graph()

        # For plotting
        self.fig = None
        if n_states == 2:
            self.cmap = colors.ListedColormap(["black", "white"])
        elif n_states > 2:
            self.cmap = cm.Set3
        elif n_states == "continuous":
            self.cmap = cm.autumn

    def get_graph(self):
        raise NotImplementedError

    def step(self, state: np.ndarray):
        densities = self.graph.a @ state
        lo, hi = self.mu - self.sigma, self.mu + self.sigma
        switch = (densities < lo) | (densities > hi)

        new_state = state.copy()
        new_state[switch] = 1 - state[switch]

        return new_state

    def evolve(self, initial_state: np.ndarray, steps: int = 1):
        states = [initial_state]
        for _ in range(steps):
            states.append(self.step(states[-1]))

        return np.array(states)

    def plot(self, state):
        raise NotImplementedError


class VoronoiCA(CA):
    def get_graph(self):
        while True:
            try:
                points = np.random.rand(self.n_cells, 2)
                graph = Graph(x=points)
                graph = Delaunay()(graph)
                assert graph.a.shape[0] == graph.a.shape[1]
                graph = NormalizeAdj(symmetric=False)(graph)
                break
            except (ValueError, AssertionError):
                pass

        return graph

    def evolve(self, initial_state: np.ndarray, steps: int = 1):
        states = [initial_state]
        for _ in range(steps):
            states.append(self.step(states[-1]))

        return np.array(states)

    def plot(self, state: np.ndarray):
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.gca()
            self.voronoi = Voronoi(self.graph.x)
            self.polygons = voronoi_utils.voronoi_polygons(self.voronoi)
            self.ax = voronoi_utils.plot_polygons(
                self.polygons, state, self.ax, self.cmap
            )

        for i, s in enumerate(state):
            self.ax.patches[i].set_facecolor(self.cmap(s))
