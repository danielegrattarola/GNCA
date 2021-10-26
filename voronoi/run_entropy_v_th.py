"""
This script plots the two entropies of the Voronoi GCA as a function of the threshold.
"""
from joblib import Parallel, delayed
from tqdm import tqdm

from modules.ca import *
from voronoi.measures import shannon_entropy, word_entropy

n_cells = 1000
steps = 1000
th_limits = (0.0, 1.0)
search_space_size = 1000
th_range = np.linspace(*th_limits, search_space_size)


# Run
def run(th, steps):
    shannon_Hs = []
    word_Hs = []
    for _ in range(5):
        state = np.random.randint(0, 2, n_cells)
        ca = VoronoiCA(n_cells, mu=0, sigma=th)
        history = ca.evolve(state, steps=steps)
        shannon_Hs.append(np.mean(shannon_entropy(history)))
        word_Hs.append(np.mean(word_entropy(history)))
    return np.mean(shannon_Hs), np.mean(word_Hs)


results = Parallel(n_jobs=-1)(delayed(run)(th, steps) for th in tqdm(th_range))
results = np.array(results)  # (search_space_size, 2)

# Plot
cmap = plt.get_cmap("Set2")
plt.figure(figsize=(3, 3))
plt.plot(th_range, results[:, 0], label=r"$H_s$", color=cmap(0))
plt.plot(th_range, results[:, 1], label=r"$H_w$", color=cmap(1))
plt.axvline(0.4, linestyle="dashed", alpha=0.5, color="#333333")
plt.xlabel(r"$\kappa$")
plt.ylabel(r"$Entropy$")
plt.legend()
plt.tight_layout()
plt.savefig("results/entropy_vs_th.pdf", bbox_inches="tight")

plt.figure(figsize=(3, 3))
plt.scatter(*results.T, c=th_range)
plt.xlabel(r"$H_s$")
plt.ylabel(r"$H_w$")
cbar = plt.colorbar()
cbar.set_label(r"$\kappa$")
plt.tight_layout()
plt.savefig("results/word_entropy_vs_shannon_entropy.pdf", bbox_inches="tight")

plt.show()
