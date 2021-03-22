import numpy as np
from swarms.rl_extensions.envs import BoidSphereEnv2D


DT = 0.3


def get_original_action(env):
    acc = []
    for agent in env._env.population:
        agent.observe(env._env)
        agent.decide(env._env.goals)
        acc.append(agent.acceleration)

    return np.vstack(acc) * DT


def combine_env_states(agent_states, obstacle_states, goal_states):
    state = np.concatenate([goal_states, obstacle_states, agent_states], axis=0)
    return np.expand_dims(state, 0)  # Add time_seg_len dim to state for swarmnet


def main():

    env = BoidSphereEnv2D(5, 1, DT)

    state = env.reset()
    sequence = [combine_env_states(*state)]

    rewards = []
    dones = []
    for _ in range(50):
        action = get_original_action(env)
        next_state, reward, done = env.step(action)

        sequence.append(combine_env_states(*next_state))
        rewards.append(reward)
        dones.append(done)

    np.save('trajectory.npy', sequence)
    # np.save('reward.npy', rewards)
    # np.save('dones.npy', dones)


if __name__ == '__main__':
    main()
