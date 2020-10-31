import random
import numpy as np
import tensorflow as tf


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def add_batch_dim(state):
    """
    Add batch dimmenstion to state.
    """
    if isinstance(state, (tuple, list)):
        return [s[np.newaxis, ...] for s in state]
    else:
        return state[np.newaxis, ...]


def system_edges(goals, obstacles, boids):
    """Edge types of the directed graph representing the influences between
    elements of the system.
        |    |Goal|Obst|Boid|
        |Goal| 0  | 0  | 1  |
        |Obst| 0  | 0  | 2  |
        |Boid| 0  | 0  | 3  |
    """
    particles = goals + obstacles + boids
    edges = np.zeros((particles, particles), dtype=int)

    up_to_goal = goals
    up_to_obs = up_to_goal + obstacles
    up_to_boids = up_to_obs + boids

    edges[0, up_to_obs:up_to_boids] = 1  # influence from goal to boid.
    edges[up_to_goal:up_to_obs, up_to_obs:up_to_boids] = 2  # influence from obstacle to boid.
    edges[up_to_obs:up_to_boids, up_to_obs:up_to_boids] = 3  # influence from boid to boid.

    np.fill_diagonal(edges, 0)
    return edges


def combine_env_states(agent_states, obstacle_states, goal_states):
    state = np.concatenate([goal_states, obstacle_states, agent_states], axis=0)
    return np.expand_dims(state, 0)  # Add time_seg_len dim to state for swarmnet


def pad_data(data, pad_to_size, dims):
    # Pad 0s before the goal
    pad_shape = [(pad_to_size - s if i in dims else 0, 0) for i, s in enumerate(data.shape)]
    padded_data = np.pad(data, pad_shape, mode='constant', constant_values=0)
    return padded_data


def get_mask(size, max_size):
    mask = np.zeros(max_size)
    mask[-size:] = 1
    return np.expand_dims(mask, -1)
