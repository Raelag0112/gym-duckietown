import ast
import argparse
import logging

import os
import numpy as np

# Duckietown Specific
from reinforcement.pytorch.ddpg import DDPG
from utils.env import launch_env
from utils.wrappers import NormalizeWrapper, GrayscaleWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
from gym.wrappers import FrameStack

def _enjoy(args):
    # Launch the env with our helper function
    env = launch_env()
    print("Initialized environment")

    # Wrappers
    env = ResizeWrapper(env)
    env = GrayscaleWrapper(env)
    env = NormalizeWrapper(env)
    env = FrameStack(env, 3)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
    policy.load(filename=args.policy, directory='reinforcement/pytorch/models/')

    obs = env.reset()
    done = False

    while True:
        while not done:
            action = policy.predict(np.array(obs))
            # Perform action
            obs, reward, done, _ = env.step(action)
            env.render()
        done = False
        obs = env.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', default='ddpg', help='Name of the initial policy')
    _enjoy(parser.parse_args())
