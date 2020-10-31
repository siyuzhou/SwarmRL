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

MIN_NUM_BOIDS = 1
MAX_NUM_BOIDS = 5
MIN_NUM_SPHERES = 1
MAX_NUM_SPHERES = 3
NUM_GOALS = 1
MAX_NUM_NODES = MAX_NUM_BOIDS + MAX_NUM_SPHERES + NUM_GOALS
DT = 0.3

BOID_SIZE = 2
SPHERE_SIZE = 7

ACTION_BOUND = 5. * DT

ROLLOUT_STEPS = 8
TRAIN_FREQUENCY = 4096
T_MAX = 60


# def set_init_weights(model):
#     init_weights = [weights / 10 for weights in model.get_weights()]
#     model.set_weights(init_weights)


def get_swarmnet_actorcritic(params, log_dir=None):
    swarmnet, inputs = SwarmNet.build_model(
        MAX_NUM_NODES, 2*NDIM, params, return_inputs=True)

    if log_dir:
        load_model(swarmnet, log_dir)

    # Action from SwarmNetW
    actions = swarmnet.out_layer.output[:, :, NDIM:]

    # Value from SwarmNet
    encodings = swarmnet.graph_conv.output[:, :, :]

    value_function = MLP([64, 64, 1], activation=None, name='value_function')
    values = value_function(encodings)  # shape [batch, NUM_GOALS+MAX_NUM_SPHERES+MAX_NUM_BOIDS, 1]

    actorcritic = tf.keras.Model(inputs=inputs,
                                 outputs=[actions, values],
                                 name='SwarmnetActorcritic')

    # Register non-overlapping `actor` and `value_function` layers for fine control
    # over trainable_variables
    actorcritic.encoding = swarmnet.graph_conv
    actorcritic.actor = swarmnet.out_layer
    actorcritic.critic = value_function

    return actorcritic


def pretrain_value_function(agent, env, stop_at_done=True):
    # Form edges as part of inputs to swarmnet.
    edges = utils.system_edges(NUM_GOALS, NUM_SPHERES, NUM_BOIDS)
    edge_types = one_hot(edges, EDGE_TYPES)

    agent.model.encoding.trainable = False
    step = 0
    for episode in range(ARGS.epochs):
        state = env.reset()
        state = utils.combine_env_states(*state)

        reward_episode = 0
        for t in range(T_MAX):
            action, log_prob = agent.act([state, edge_types])

            # Ignore "actions" from goals and obstacles.
            next_state, reward, done = env.step(action)
            # reward = combine_env_rewards(*reward)
            next_state = utils.combine_env_states(*next_state)

            agent.store_transition([state, edge_types], action, reward,
                                   log_prob, [next_state, edge_types], done)

            state = next_state
            reward_episode += np.sum(reward)

            # Overide done if non_stop is True:
            done &= stop_at_done
            step += 1
            if done or (t == T_MAX-1):
                agent.finish_rollout([state, edge_types], done)
            if step % TRAIN_FREQUENCY == 0:
                agent.update(ARGS.batch_size, actor_steps=0)
            if done:
                break

        print(f'\r Episode {episode} | Reward {reward_episode:8.2f} | End t = {t} ',
              end='')
        if (episode + 1) % 100 == 0:
            print('')
            save_model(agent.model, ARGS.log_dir+'/rl')


def train(agent):
    # Fix goal-agent edge function
    goal_edge = agent.model.encoding.edge_encoder.edge_encoders[0]
    if ARGS.mode > 0:
        goal_edge.trainable = False

    reward_all_episodes = []
    ts = []
    step = 0
    for episode in range(ARGS.epochs):
        # Initialize num_boids and num_spheres.
        num_boids = np.random.randint(MIN_NUM_BOIDS, MAX_NUM_BOIDS+1)
        num_spheres = np.random.randint(MIN_NUM_SPHERES, MAX_NUM_SPHERES+1)
        # Create mask.
        mask = utils.get_mask(num_boids, MAX_NUM_NODES)
        # Form edges as part of inputs to swarmnet.
        edges = utils.system_edges(NUM_GOALS, num_spheres, num_boids)
        edge_types = one_hot(edges, EDGE_TYPES)
        padded_edge_types = utils.pad_data(edge_types, MAX_NUM_NODES, dims=[0, 1])

        env = BoidSphereEnv2D(num_boids, num_spheres, NUM_GOALS, DT,
                              boid_size=BOID_SIZE, sphere_size=SPHERE_SIZE)
        state = env.reset()  # When seed is provided, env is essentially fixed.
        state = utils.combine_env_states(*state)  # Shape [1, NUM_GOALS+num_spahers+num_boids, 4]

        reward_episode = 0
        for t in range(T_MAX):
            padded_state = utils.pad_data(state, MAX_NUM_NODES, [1])
            padded_action, padded_log_prob = agent.act(
                [padded_state, padded_edge_types], mask, training=True)

            # Ignore "actions" from goals and obstacles.
            action = padded_action[-num_boids:, :]

            next_state, reward, done = env.step(action)
            next_state = utils.combine_env_states(*next_state)

            padded_next_state = utils.pad_data(next_state, MAX_NUM_NODES, [1])
            padded_reward = utils.pad_data(reward, MAX_NUM_NODES, [0])

            agent.store_transition([padded_state, padded_edge_types], padded_action, padded_reward,
                                   padded_log_prob, [padded_next_state, padded_edge_types], done, mask)

            state = next_state
            reward_episode += np.sum(reward)

            step += 1
            if done or (t == T_MAX-1):
                agent.finish_rollout([padded_state, padded_edge_types], done, mask)

            if step % TRAIN_FREQUENCY == 0:
                agent.update(ARGS.batch_size)
            if done:
                break

        env.close()

        ts.append(t)
        reward_all_episodes.append(reward_episode)

        # Log to tensorboard
        with agent.summary_writer.as_default():
            tf.summary.scalar('Episode Reward', reward_episode, step=episode)
            tf.summary.scalar('Terminal Timestep', t, step=episode)

        print(f'\r Episode {episode} | Reward {reward_episode:8.2f} | ' +
              f'Avg. R {np.mean(reward_all_episodes[-100:]):8.2f} | Avg. End t = {np.mean(ts[-100:]):3.0f}',
              end='')
        if (episode + 1) % 1000 == 0:
            print('')
            # Hack for preserving the order or weights while saving
            goal_edge.trainable = True
            save_model(agent.model, ARGS.log_dir+'/rl')
            save_model(agent.model, ARGS.log_dir+f'/rl_{episode}')
            if ARGS.mode > 0:
                goal_edge.trainable = False
            np.save(ARGS.log_dir+'/rl/train_rewards.npy', reward_all_episodes)
            np.save(ARGS.log_dir+'/rl/terminal_ts.npy', ts)

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

    if ARGS.gif:
        import imageio
        frames = []

    for t in range(ARGS.steps):
        action, _ = agent.act([state, edge_types])
        value = agent.value([state, edge_types])

        print(f'Step {t}')
        print('Action', action)
        # print(test_out)

        # Ignore "actions" from goals and obstacles.
        next_state, reward, done = env.step(action)

        if ARGS.gif:
            frames.append(env.render())
        else:
            env.render()
        # reward = combine_env_rewards(*reward)

        state = utils.combine_env_states(*next_state)

        reward_sequence.append(reward)
        trajectory.append(state)

        if done:
            break

    print(f' Final Reward {np.sum(reward_sequence)} | End t = {t}')
    np.save(os.path.join(ARGS.log_dir, 'test_trajectory.npy'), trajectory)
    np.save(os.path.join(ARGS.log_dir, 'reward_sequence.npy'), reward_sequence)
    if ARGS.gif:
        print('Saving GIF...')
        imageio.mimsave(os.path.join(ARGS.log_dir, ARGS.gif + '.gif'), frames, fps=6)


def main():
    # Tensorboard logging setup
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(
        ARGS.log_dir + '/' + current_time) if ARGS.train or ARGS.pretrain else None

    swarmnet_params = load_model_params(ARGS.config)

    actorcritic = get_swarmnet_actorcritic(swarmnet_params, ARGS.log_dir)
    # NOTE: lock node_updater layer and final dense layer.
    if ARGS.mode == 2:
        actorcritic.encoding.node_decoder.trainable = False
        actorcritic.actor.trainable = False

    # Load weights trained from RL.
    rl_log = os.path.join(ARGS.log_dir, 'rl')
    if os.path.exists(rl_log):
        load_model(actorcritic, rl_log)
    # elif not os.path.exists(os.path.join(ARGS.log_dir, 'weights.h5')):
    #     print('Re-initialize weights.')
    #     set_init_weights(actorcritic)

    swarmnet_agent = PPOAgent(actorcritic, NDIM,
                              action_bound=None,
                              rollout_steps=ROLLOUT_STEPS,
                              memory_capacity=4096,
                              summary_writer=summary_writer,
                              mode=ARGS.mode)

    if ARGS.pretrain:
        pretrain_value_function(swarmnet_agent)
    elif ARGS.train:
        train(swarmnet_agent)
    elif ARGS.test:
        test(swarmnet_agent)


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
    parser.add_argument("--gif", type=str, default=None,
                        help="store output as gif with the given filename")
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--steps", type=int, default=T_MAX,
                        help='max timestep per episode')
    ARGS = parser.parse_args()

    ARGS.config = os.path.expanduser(ARGS.config)
    ARGS.log_dir = os.path.expanduser(ARGS.log_dir)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    utils.set_seed(ARGS.seed)
    main()
