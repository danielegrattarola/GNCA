"""
Shows an animation of the Boids system.
"""
from modules.boids import Boids

boids = Boids(
    min_speed=0.0001,  # Min speed of the boids
    max_speed=0.01,  # Max speed of the boids
    max_force=0.1,  # Max amount of steering that any single update is allowed to add
    max_turn=5,  # How many degrees is a boid allowed to turn
    perception=0.25,  # How distant must two boids be in order to be neighbors
    crowding=0.025,  # How much groups are pushed apart (lower = tighter groups)
    n_boids=100,  # How many boids in the environment
    dt=1,  # Size of a time step (lower = more precise simulation)
    canvas_scale=1,  # Canvas is rescaled by this amount (used to control size)
    boundary_size_pctg=0.2,  # Relative size of the soft boundary
    wrap=False,  # If True, wrap around instead of avoiding boundary
    show=True,  # Show an animated plot of the boids everytime update_boids is called
)
boids.generate_trajectory(1000)
