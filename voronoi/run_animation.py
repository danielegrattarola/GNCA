"""
This script shows an animation of the Voronoi GCA.
"""
import matplotlib.pyplot as plt
import numpy as np

from modules.ca import VoronoiCA

n_cells = 1000
mu = 0.0
sigma = 0.42
steps = 1000


# Run
initial_state = np.random.randint(0, 2, n_cells)

ca = VoronoiCA(n_cells, mu=mu, sigma=sigma)
history = ca.evolve(initial_state, steps=steps)

# Animation
plt.ion()
plt.show()
for state in history:
    ca.plot(state)
    plt.draw()
    plt.pause(0.1)
