import os
import time
import gym
import numpy as np
from swarmnet.utils import load_model_params
from multiprocessing_env import SubprocVecEnv
# from stable_baselines.common import vec_env
from run_rl import ROLLOUT_STEPS, get_swarmnet_actorcritic
from swarmnet.utils import save_model, load_model, one_hot, load_model_params
from ppo_agent import PPOAgent
import utils


NDIM = 2
EDGE_TYPES = 4

DT = 0.3
NUM_ENVS = 8
PPO_STEPS = 128
ENV_ID = 'swarms:BoidSphere2D-v0'

MIN_NUM_BOIDS = 1
MAX_NUM_BOIDS = 5
MIN_NUM_SPHERES = 1
MAX_NUM_SPHERES = 3
NUM_GOALS = 1
MAX_NUM_NODES = MAX_NUM_BOIDS + MAX_NUM_SPHERES + NUM_GOALS
DT = 0.3

BOID_SIZE = 2
SPHERE_SIZE = 7

ROLLOUT_STEPS = 8


def make_env(num_boids, num_spheres):
    # returns a function which creates a single environment
    def _thunk():
        env = gym.make(ENV_ID,
                       num_boids=num_boids, num_obstacles=num_spheres, num_goals=NUM_GOALS, dt=DT,
                       boid_size=BOID_SIZE, sphere_size=SPHERE_SIZE)

        return env
    return _thunk


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    start_t = time.time()

    envs, env_num_boids = [], []
    padded_edge_types, masks = [], []
    for _ in range(NUM_ENVS):
        num_boids = np.random.randint(MIN_NUM_BOIDS, MAX_NUM_BOIDS + 1)
        num_spheres = np.random.randint(MIN_NUM_SPHERES, MAX_NUM_SPHERES + 1)

        env_num_boids.append(num_boids)

        envs.append(make_env(num_boids, num_spheres))

        edges = utils.system_edges(NUM_GOALS, num_spheres, num_boids)
        edge_types = one_hot(edges, EDGE_TYPES)
        padded_edge_types.append(utils.pad_data(edge_types, MAX_NUM_NODES, dims=[0, 1]))

        mask = utils.get_mask(num_boids, MAX_NUM_NODES)
        masks.append(mask)

    envs = SubprocVecEnv(envs)
    padded_edge_types = np.array(padded_edge_types)
    masks = np.array(masks)

    swarmnet_params = load_model_params('config/il_rl.json')
    actorcritic = get_swarmnet_actorcritic(swarmnet_params, '../../Logs/swarmnet_rl_test')

    swarmnet_agent = PPOAgent(actorcritic, NDIM,
                              action_bound=None,
                              rollout_steps=ROLLOUT_STEPS,
                              memory_capacity=4096,
                              summary_writer=None,
                              mode=0)

    states = envs.reset()
    states = [utils.combine_env_states(*state) for state in states]
    padded_states = np.array([utils.pad_data(state, MAX_NUM_NODES, [1]) for state in states])

    for step in range(PPO_STEPS):
        if (step + 1) % 10 == 0:
            print('Step', step)

        padded_actions = swarmnet_agent.act_batch([padded_states, padded_edge_types], masks)[0]
        next_states, rewards, dones, infos = envs.step(
            [padded_action[-num_boid:, :]
             for padded_action, num_boid in zip(padded_actions, env_num_boids)])

        padded_states = np.array([utils.pad_data(utils.combine_env_states(*state), MAX_NUM_NODES, [1])
                                  for state in next_states])

    end_t = time.time()

    print("Time spent", end_t - start_t)
