import numpy as np
from swarms.rl_extensions.envs import BoidSphereEnv2D


def get_original_action(env):
    acc = []
    for agent in env._env.population:
        acc.append(agent.acceleration)

    return np.vstack(acc)


def main():

    env = BoidSphereEnv2D(5, 1, 1, 0.3)

    boid_state, _, _ = env.reset()
    sequence = [boid_state]

    rewards = []
    dones = []
    for _ in range(50):
        action = get_original_action(env)
        next_state, reward, done = env.step(action * 0.3)

        boid_state, _, _ = next_state

        sequence.append(boid_state)
        rewards.append(reward)
        dones.append(done)

    # np.save('trajectory.npy', sequence)
    # np.save('reward.npy', rewards)
    # np.save('dones.npy', dones)


if __name__ == '__main__':
    main()
