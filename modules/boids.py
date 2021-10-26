import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from spektral import utils
from spektral.data import Dataset, Graph
from tqdm import tqdm


class Boids:
    def __init__(
        self,
        min_speed=0.0001,  # Min speed of the boids
        max_speed=0.01,  # Max speed of the boids
        max_force=0.1,  # Max amount of steering that any single update is allowed to add
        max_turn=5,  # How many degrees is a boid allowed to turn
        perception=0.15,  # How distant must two boids be in order to be neighbors
        crowding=0.015,  # How much groups are pushed apart (lower = tighter groups)
        n_boids=100,  # How many boids in the environment
        dt=1,  # Size of a time step (lower = more precise simulation)
        canvas_scale=1,  # Canvas is rescaled by this amount (used to control size)
        boundary_size_pctg=0.2,  # Relative size of the soft boundary
        wrap=False,  # If True, wrap around instead of avoiding boundary
        limits=True,  # If True, enforce speed and turn limits
        show=False,  # Show an animated plot of the boid everytime update_boids is called
    ):
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.max_force = max_force
        self.max_turn = max_turn
        self.perception = perception
        self.crowding = crowding
        self.n_boids = n_boids
        self.dt = dt
        self.canvas_scale = canvas_scale
        self.boundary_size_pctg = boundary_size_pctg
        self.wrap = wrap
        self.limits = limits

        self.borders = canvas_scale * np.array([-1, -1, 1, 1])  # Hard borders of canvas
        self.center = (self.borders[2:] + self.borders[:2]) / 2  # Center of the canvas

        # Soft boundary inside which boids are pushed towards the center to avoid
        # leaving the canvas
        self.boundary_margins = self.borders * boundary_size_pctg
        self.boundaries = self.borders - self.boundary_margins

        self.show = show
        self.figure = None

    def update_boids(self, positions, velocities, return_accel=False):
        accelerations = np.zeros_like(velocities)

        if self.wrap:
            positions = ((positions + 1) % 2) - 1  # Wrap around
        else:
            accelerations += self.avoid_borders(positions)  # Avoid edge collisions

        neighbors = self.get_neighbors(positions)
        accelerations += self.get_separation(neighbors, positions)
        accelerations += self.get_alignment(neighbors, velocities) / 8
        accelerations += self.get_cohesion(neighbors, positions) / 100

        velocities_new = velocities + accelerations * self.dt
        if self.limits:
            velocities = self.enforce_limits(velocities, velocities_new)
        else:
            velocities = velocities_new

        # Update positions
        positions = positions + velocities * self.dt

        # positions[:, 0] = np.clip(positions[:, 0], self.borders[0], self.borders[2])
        # positions[:, 1] = np.clip(positions[:, 1], self.borders[1], self.borders[3])

        # Plot if needed
        if self.show:
            self.plot(positions)

        return (positions, velocities, neighbors) + (
            (accelerations,) if return_accel else ()
        )

    def generate_trajectory(self, steps, init=None, return_accel=False):
        if init is None:
            positions, velocities, neighbors = self.get_random_init(self.n_boids)
        else:
            assert (
                len(init) == 2
            ), "Expected init to have lenght 2 (positions, velocities)"
            positions, velocities = init
            neighbors = self.get_neighbors(positions)
        history = {
            "positions": [positions],
            "velocities": [velocities],
            "neighbors": [neighbors],
        }
        if return_accel:
            history["accelerations"] = []
        for _ in range(steps):
            output = self.update_boids(positions, velocities, return_accel=return_accel)
            positions, velocities, neighbors = output[:3]
            history["positions"].append(positions)
            history["velocities"].append(velocities)
            history["neighbors"].append(neighbors)
            if return_accel:
                history["accelerations"].append(output[3])

        history["positions"] = np.array(history["positions"])
        history["velocities"] = np.array(history["velocities"])
        if return_accel:
            history["accelerations"] = np.array(history["accelerations"])

        return history

    def avoid_borders(self, positions):
        """If a boid is within the external margins, steer it towards the centre"""
        in_margin = np.any(positions < self.boundaries[:2], -1) | np.any(
            positions > self.boundaries[2:], -1
        )
        steering = np.zeros_like(positions)

        steering[in_margin] += self.center - positions[in_margin]

        return steering

    def get_neighbors(self, positions):
        neighbors = (
            np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
            < self.perception
        )

        neighbors = sp.coo_matrix(neighbors, dtype=int)
        neighbors.setdiag(0)

        return neighbors

    def get_separation(self, neighbors, positions):
        """
        Get the steering component to separate boids that are too close
        """
        self_idx, neig_idx = neighbors.row, neighbors.col
        distances = np.linalg.norm(positions[self_idx] - positions[neig_idx], axis=-1)
        mask = distances < self.crowding
        steering = np.zeros_like(positions)
        changes = -(positions[neig_idx[mask]] - positions[self_idx[mask]])
        np.add.at(steering, self_idx[mask], changes)
        steering = self.clamp(steering)

        return steering

    def get_alignment(self, neighbors, velocities):
        """
        Get the steering component to align the velocities of neighbors
        """
        degree = utils.degree_power(neighbors, -1.0)
        steering = degree @ neighbors @ velocities
        steering -= velocities
        steering = self.clamp(steering)

        return steering

    def get_cohesion(self, neighbors, positions):
        """
        Get the steering component to align the positions of neighbors
        """
        return self.get_alignment(neighbors, positions)

    def enforce_limits(self, velocities_old, velocities_new):
        # Update velocities
        velocities_old_polar = to_polar(velocities_old)
        velocities_new_polar = to_polar(velocities_new)

        # Enforce turn limit
        phi_diff = (
            180 - (180 - velocities_new_polar[:, 1] + velocities_old_polar[:, 1]) % 360
        )
        mask = np.abs(phi_diff) > self.max_turn
        velocities_new_polar[mask, 1] = (
            velocities_old_polar[mask, 1] + np.sign(phi_diff[mask]) * self.max_turn
        )
        velocities = to_cartesian(velocities_new_polar)

        # Enforce speed limit
        speed = np.linalg.norm(velocities, axis=-1)
        velocities[speed < self.min_speed] = scale(
            velocities[speed < self.min_speed], self.min_speed
        )
        velocities[speed > self.max_speed] = scale(
            velocities[speed > self.max_speed], self.max_speed
        )

        return velocities

    def clamp(self, force):
        """
        Clamp a given force (steering) to the maximum value allowed (to make things
        more stable)
        """
        # to_clamp = np.linalg.norm(force, axis=-1) > self.max_force
        # force[to_clamp] = scale(force[to_clamp], lenght=self.max_force)

        return force

    def get_random_init(self, n_boids):
        """
        Get a random initial position and velocity for each boid
        :param n_boids: int, number of boids
        """
        positions = np.stack(
            [
                np.random.uniform(*self.boundaries[::2], n_boids),
                np.random.uniform(*self.boundaries[1::2], n_boids),
            ]
        ).T
        velocities = to_cartesian(
            np.random.uniform(-1, 1, (n_boids, 2)) * self.max_speed
        )
        neighbors = self.get_neighbors(positions)

        return positions, velocities, neighbors

    def plot(self, positions, **kwargs):
        if self.figure is None:
            plt.ion()
            self.figure = plt.figure()
            axes = plt.axes(xlim=self.borders[::2], ylim=self.borders[1::2])
            self.scatter = axes.scatter(
                positions[:, 0],
                positions[:, 1],
                marker=".",
                edgecolor="k",
                lw=0.5,
                **kwargs
            )
            # anim = animation.FuncAnimation(figure, animate, frames=50, interval=1)
            plt.show()
        self.scatter.set_offsets(positions)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


class BoidsDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.graphs = dataset

    def read(self):
        return []


def to_polar(cartesian_coords):
    x, y = cartesian_coords.T
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x) * 180 / np.pi
    return np.stack((rho, phi), -1)


def to_cartesian(polar_coords):
    rho, phi = polar_coords.T
    phi *= np.pi / 180
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.stack((x, y), -1)


def scale(x, lenght=1.0):
    return lenght * x / np.linalg.norm(x, axis=-1, keepdims=True)


def history_to_samples(history, accel=False):
    inputs = np.concatenate((history["positions"], history["velocities"]), axis=-1)
    neighbors = history["neighbors"]
    if accel and "accelerations" in history:
        targets = history["accelerations"]
    else:
        targets = inputs[1:]

    return [(x, a, y) for x, a, y in zip(inputs[:-1], neighbors[:-1], targets)]


def make_dataset(reps, steps, return_boids=False, accel=False, **kwargs):
    init = kwargs.pop("init", None)
    n_jobs = kwargs.pop("n_jobs", 1)
    boids = Boids(**kwargs)

    histories = Parallel(n_jobs=n_jobs)(
        delayed(boids.generate_trajectory)(steps, init=init, return_accel=accel)
        for _ in tqdm(range(reps))
    )

    samples = []
    for history in histories:
        samples.extend(history_to_samples(history, accel=accel))

    graphs = [Graph(x=x, a=a, y=y) for x, a, y in samples]
    dataset = BoidsDataset(graphs)

    if return_boids:
        return dataset, boids
    else:
        return dataset
