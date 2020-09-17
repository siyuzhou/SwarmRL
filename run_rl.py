import os
import argparse
import datetime

import tensorflow as tf
import numpy as np

from swarmnet import SwarmNet
from swarmnet.modules import MLP
from swarmnet.utils import save_model, load_model, one_hot, load_model_params

import utils
from ppo_agent import PPOAgent
from swarms.rl_extensions.envs import BoidSphereEnv2D

NDIM = 2
EDGE_TYPES = 4

NUM_BOIDS = 5
NUM_SPHERES = 0
NUM_GOALS = 1
DT = 0.3

BOID_SIZE = 2
SPHERE_SIZE = 6

ACTION_BOUND = 5. * DT

ROLLOUT_STEPS = 8
T_MAX = 60


def set_init_weights(model):
    init_weights = [weights / 10 for weights in model.get_weights()]
    model.set_weights(init_weights)


def get_swarmnet_actorcritic(params, log_dir=None):
    swarmnet, inputs = SwarmNet.build_model(
        NUM_GOALS+NUM_SPHERES+NUM_BOIDS, 2*NDIM, params, return_inputs=True)

    if log_dir:
        load_model(swarmnet, log_dir)

    # Action from SwarmNetW
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
    edges = utils.system_edges(NUM_GOALS, NUM_SPHERES, NUM_BOIDS)
    edge_types = one_hot(edges, EDGE_TYPES)

    agent.model.encoding.trainable = False

    for episode in range(ARGS.epochs):
        state = env.reset()
        state = utils.combine_env_states(*state)

        reward_episode = 0
        for t in range(T_MAX):
            action, log_prob = agent.act([state, edge_types])

            # Ignore "actions" from goals and obstacles.
            next_state, reward, done = env.step(action)
            # reward = combine_env_rewards(*reward)

            agent.store_transition([state, edge_types], action, reward, log_prob)

            state = utils.combine_env_states(*next_state)
            reward_episode += np.sum(reward)

            # Overide done if non_stop is True:
            done &= stop_at_done

            if (len(agent.rollout_buffer) >= ARGS.batch_size) or done or (t == T_MAX-1):
                agent.finish_rollout([state, edge_types], done)
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
    # Fix goal-agent edge function
    goal_edge = agent.model.encoding.edge_encoder.edge_encoders[0]
    goal_edge.trainable = False

    # Form edges as part of inputs to swarmnet.
    edges = utils.system_edges(NUM_GOALS, NUM_SPHERES, NUM_BOIDS)
    edge_types = one_hot(edges, EDGE_TYPES)

    reward_all_episodes = []
    ts = []
    for episode in range(ARGS.epochs):
        state = env.reset() # When seed is provided, env is essentially fixed.
        state = utils.combine_env_states(*state)
        reward_episode = 0
        for t in range(T_MAX):
            action, log_prob = agent.act([state, edge_types], training=True)

            # Ignore "actions" from goals and obstacles.
            next_state, reward, done = env.step(action)
            # reward = combine_env_rewards(*reward)

            agent.store_transition([state, edge_types], action, reward, log_prob)

            state = utils.combine_env_states(*next_state)
            reward_episode += np.sum(reward)

            if (len(agent.rollout_buffer) >= ARGS.batch_size) or done or (t == T_MAX-1):
                agent.finish_rollout([state, edge_types], done)

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
        if (episode + 1) % 1000 == 0:
            print('')
            # Hack for preserving the order or weights while saving
            goal_edge.trainable = True
            save_model(agent.model, ARGS.log_dir+f'/rl_{episode}')
            goal_edge.trainable = False

    goal_edge.trainable = True
    save_model(agent.model, ARGS.log_dir+'/rl')


def test(agent, env):
    # Form edges as part of inputs to swarmnet.
    edges = utils.system_edges(NUM_GOALS, NUM_SPHERES, NUM_BOIDS)
    edge_types = one_hot(edges, EDGE_TYPES)

    state = env.reset(ARGS.seed)
    state = utils.combine_env_states(*state)
    reward_sequence = []
    trajectory = [state]
    for t in range(T_MAX):
        action, _ = agent.act([state, edge_types])
        value = agent.value([state, edge_types])

        print(f'Step {t}')
        print('Action', action, '\nValue', value)
        # print(test_out)

        # Ignore "actions" from goals and obstacles.
        next_state, reward, done = env.step(action)

        state = utils.combine_env_states(*next_state)
        
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
    # NOTE: lock node_updater layer and final dense layer.
    actorcritic.encoding.node_decoder.trainable = False
    actorcritic.actor.trainable = False

    # Load weights trained from RL.
    rl_log = os.path.join(ARGS.log_dir, 'rl')
    if os.path.exists(rl_log):
        load_model(actorcritic, rl_log)
    else:
        set_init_weights(actorcritic)
        print('Use init weights')
    
    swarmnet_agent = PPOAgent(actorcritic, NDIM, 
                              action_bound=None,
                              rollout_steps=ROLLOUT_STEPS,
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
    parser.add_argument('--batch-size', type=int, default=4096,
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

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    utils.set_seed(ARGS.seed)
    main()
