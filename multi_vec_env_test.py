# from swarms.rl_extensions.envs import BoidSphereEnv2D
import gym
from multiprocessing_env import SubprocVecEnv
# from stable_baselines.common import vec_env

DT = 0.3
NUM_ENVS = 8
PPO_STEPS = 1000
ENV_ID = 'swarms:swarms-v0'

def make_env():
    # returns a function which creates a single environment
    def _thunk():
        env = gym.make(ENV_ID)
        return env
    return _thunk

if __name__ == "__main__":
    envs = [make_env() for i in range(NUM_ENVS)]
    envs = SubprocVecEnv(envs)
    env = gym.make(ENV_ID)

    for _ in range(PPO_STEPS):
        next_state, reward, done, info = envs.step([env.action_space.sample() for _ in range(NUM_ENVS)])
        print(done)