import os
import argparse
import datetime

import tensorflow as tf
import numpy as np

from swarmnet import SwarmNet
from swarmnet.modules import MLP
from swarmnet.utils import save_model, load_model, one_hot, load_model_params

from utils import set_seed
from ppo_agent import PPOAgent
from swarms.rl_extensions.envs import BoidSphereEnv2D

NDIM = 2
EDGE_TYPES = 4

NUM_BOIDS = 1
NUM_SPHERES = 0
NUM_GOALS = 1
DT = 0.3

BOID_SIZE = 1
SPHERE_SIZE = 4

ACTION_BOUND = 5. * DT

ROLLOUT_STEPS = 8
T_MAX = 50


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
    assert state.shape == (NUM_GOALS + NUM_SPHERES + NUM_BOIDS, 2 * NDIM)
    return np.expand_dims(state, 0)  # Add time_seg_len dim to state for swarmnet


# def combine_env_rewards(agent_reward, obstacle_reward, goal_reward):
#     reward = np.concatenate([goal_reward, obstacle_reward, agent_reward], axis=-1)
#     assert reward.shape == (NUM_GOALS + NUM_SPHERES + NUM_BOIDS,)
#     return reward

def set_init_weights(model):
    init_weights = [weights / 100 for weights in model.get_weights()]
    model.set_weights(init_weights)


def get_swarmnet_actorcritic(params, log_dir=None):
    swarmnet, inputs = SwarmNet.build_model(
        NUM_GOALS+NUM_SPHERES+NUM_BOIDS, 2*NDIM, params, return_inputs=True)

    if log_dir:
        load_model(swarmnet, log_dir)

    # Action from SwarmNet
    actions = swarmnet.dense.output[:, -NUM_BOIDS:, NDIM:]
    # NOTE: Add tanh for action bound
    actions = tf.keras.activations.tanh(actions) * ACTION_BOUND

    # Value from SwarmNet
    encodings = swarmnet.graph_conv.output[:, -NUM_BOIDS:, :]

    value_function = MLP([64, 64, 1], activation=None, name='value_function')
    values = value_function(encodings) # shape [batch, NUM_BOIDS, 1]

    actorcritic = tf.keras.Model(inputs=inputs,
                                 outputs=[actions, values],
                                 name='SwarmnetActorcritic')

    # Register non-overlapping `actor` and `value_function` layers for fine control
    # over trainable_variables
    actorcritic.encoding = swarmnet.graph_conv
    actorcritic.actor = swarmnet.dense
    actorcritic.critic = value_function

    return actorcritic


def pretrain_value_function(agent, env, stop_at_done=True):
    # Form edges as part of inputs to swarmnet.
    edges = system_edges(NUM_GOALS, NUM_SPHERES, NUM_BOIDS)
    edge_types = one_hot(edges, EDGE_TYPES)

    agent.model.encoding.trainable = False

    for episode in range(ARGS.epochs):
        state = env.reset()
        state = combine_env_states(*state)

        reward_episode = 0
        for t in range(T_MAX):
            action, _ = agent.act(
                [state[np.newaxis, ...], edge_types[np.newaxis, ...]])

            # Ignore "actions" from goals and obstacles.
            next_state, reward, done = env.step(action)
            # reward = combine_env_rewards(*reward)

            agent.store_transition([state, edge_types], action, reward)

            state = combine_env_states(*next_state)
            reward_episode += np.sum(reward)

            # Overide done if non_stop is True:
            done &= stop_at_done

            if (len(agent.rollout_buffer) >= ARGS.batch_size) or done or (t == T_MAX-1):
                agent.finish_rollout(
                    [state[np.newaxis, ...], edge_types[np.newaxis, ...]], done)
            if len(agent.rollout_buffer) >= ARGS.batch_size:
                agent.update(actor_steps=0)
            if done:
                break

        print(f'\r Episode {episode} | Reward {reward_episode:8.2f} | End t = {t} ',
              end='')
        if (episode + 1) % 100 == 0:
            print('')
            save_model(agent.model, ARGS.log_dir+'/rl')


def train(agent, env):
    # Form edges as part of inputs to swarmnet.
    edges = system_edges(NUM_GOALS, NUM_SPHERES, NUM_BOIDS)
    edge_types = one_hot(edges, EDGE_TYPES)

    reward_all_episodes = []
    ts = []
    for episode in range(ARGS.epochs):
        state = env.reset()
        state = combine_env_states(*state)
        reward_episode = 0
        for t in range(T_MAX):
            action, log_prob = agent.act(
                [state[np.newaxis, ...], edge_types[np.newaxis, ...]], training=True)

            # Ignore "actions" from goals and obstacles.
            next_state, reward, done = env.step(action)
            # reward = combine_env_rewards(*reward)

            agent.store_transition([state, edge_types], action, reward, log_prob)

            state = combine_env_states(*next_state)
            reward_episode += np.sum(reward)

            if (len(agent.rollout_buffer) >= ARGS.batch_size) or done or (t == T_MAX-1):
                agent.finish_rollout(
                    [state[np.newaxis, ...], edge_types[np.newaxis, ...]], done)

            if len(agent.rollout_buffer) >= ARGS.batch_size:
                agent.update()
            if done:
                break

        ts.append(t)
        reward_all_episodes.append(reward_episode)

        # Log to tensorboard
        with agent.summary_writer.as_default():
            tf.summary.scalar('Episode Reward', reward_episode, step=episode)

        print(f'\r Episode {episode} | Reward {reward_episode:8.2f} | ' +
              f'Avg. R {np.mean(reward_all_episodes[-100:]):8.2f} | Avg. End t = {np.mean(ts[-100:]):3.0f}',
              end='')
        if (episode + 1) % 100 == 0:
            print('')
            save_model(agent.model, ARGS.log_dir+'/rl')


def test(agent, env):
    # Form edges as part of inputs to swarmnet.
    edges = system_edges(NUM_GOALS, NUM_SPHERES, NUM_BOIDS)
    edge_types = one_hot(edges, EDGE_TYPES)

    state = env.reset()
    state = combine_env_states(*state)
    reward_sequence = []
    trajectory = [state]
    for t in range(T_MAX):
        action, _ = agent.act(
            [state[np.newaxis, ...], edge_types[np.newaxis, ...]])

        print(action)
        # Ignore "actions" from goals and obstacles.
        next_state, reward, done = env.step(action)
        # reward = combine_env_rewards(*reward)

        state = combine_env_states(*next_state)
        
        reward_sequence.append(reward)
        trajectory.append(state)
        if done:
            break

    print(f' Final Reward {np.sum(reward_sequence)} | End t = {t}')
    np.save(os.path.join(ARGS.log_dir, 'test_trajectory.npy'), trajectory)
    np.save(os.path.join(ARGS.log_dir, 'reward_sequence.npy'), reward_sequence)


def main():
    # Tensorboard logging setup
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(ARGS.log_dir + '/'+ current_time) if ARGS.train or ARGS.pretrain else None

    swarmnet_params = load_model_params(ARGS.config)

    actorcritic = get_swarmnet_actorcritic(swarmnet_params, ARGS.log_dir)

    # Load weights trained from RL.
    rl_log = os.path.join(ARGS.log_dir, 'rl')
    if os.path.exists(rl_log):
        load_model(actorcritic, rl_log)
    else:
        set_init_weights(actorcritic)
    
    swarmnet_agent = PPOAgent(actorcritic, NDIM, 
                             action_bound=None,
                             summary_writer=summary_writer)

    env = BoidSphereEnv2D(NUM_BOIDS, NUM_SPHERES, NUM_GOALS, DT, boid_size=BOID_SIZE, sphere_size=SPHERE_SIZE)

    if ARGS.pretrain:
        pretrain_value_function(swarmnet_agent, env)
    elif ARGS.train:
        train(swarmnet_agent, env)
    elif ARGS.test:
        test(swarmnet_agent, env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='model config file')
    parser.add_argument('--log-dir', type=str,
                        help='log directory')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training steps')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--pretrain', action='store_true', default=False,
                        help='turn on pretraining of value function')
    parser.add_argument('--train', action='store_true', default=False,
                        help='turn on training')
    parser.add_argument('--test', action='store_true', default=False,
                        help='turn on test')
    parser.add_argument('--seed', type=int, default=1337,
                        help='set random seed')
    ARGS = parser.parse_args()

    ARGS.config = os.path.expanduser(ARGS.config)
    ARGS.log_dir = os.path.expanduser(ARGS.log_dir)

    set_seed(ARGS.seed)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    main()
