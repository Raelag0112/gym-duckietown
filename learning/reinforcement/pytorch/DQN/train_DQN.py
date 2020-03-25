import ast
import argparse
import logging
import time
import random
import math

import os
import numpy as np

# Duckietown Specific
from learning.reinforcement.pytorch.dqn import DQN
from learning.reinforcement.pytorch.utils import seed, evaluate_policy, ReplayBuffer
from learning.utils.env import launch_env
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
from gym_duckietown.wrappers import DiscreteWrapper
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _train(args):   
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    # Launch the env with our helper function
    env = launch_env()
    print("Initialized environment")

    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DiscreteWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")
    
    # Set seeds
    seed(args.seed)

    state_dim = env.observation_space.shape

    # Initialize policy
    _, screen_height, screen_width = env.observation_space.shape
    n_actions = 3
    # 3 actions are possible while using the discrete action wrapper
    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    replay_buffer = ReplayBuffer(args.replay_buffer_max_size)
    print("Initialized DQN")
    
    # Evaluate untrained policy
    #evaluations= [evaluate_policy(env, policy_net)]
    evaluations = [evaluate_policy(env, policy, eval_episodes=10, max_timesteps=env_timesteps)];
   
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    episode_reward = None
    env_counter = 0
    reward = 0
    episode_timesteps = 0
    start_time = time.time()
    print("Starting training")
    while total_timesteps < args.max_timesteps:
        
        print("timestep: {} | reward: {}".format(total_timesteps, reward))
            
        if done:
            if total_timesteps != 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f Elapsed Time: %f") % (
                    total_timesteps, episode_num, episode_timesteps, episode_reward, elapsed_time))
                for i in range(episode_timesteps):
                    policy_net.optimize_model(replay_buffer, target_net, args.batch_size, args.discount)
                    if(i%args.target_update == 0):
                        target_net.load_state_dict(policy_net.state_dict())

                # Evaluate episode
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval %= args.eval_freq
                    evaluations.append(evaluate_policy(env, policy_net))
                    print("rewards at time {}: {}".format(total_timesteps, evaluations[-1]))

                    if args.save_models:
                        policy.save(filename='ddpg', directory=args.model_dir)
                    np.savez("./results/rewards.npz",evaluations)

            # Reset environment
            env_counter += 1
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        sample = random.random()
        eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * \
            math.exp(-1. * total_timesteps / args.eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = policy_net.predict(np.array(obs))
        else:
            action = torch.tensor([random.randrange(n_actions)], device=device, dtype=torch.long)

        # Perform action
        new_obs, reward, done, _ = env.step(action)

        if episode_timesteps >= args.env_timesteps:
            done = True

        done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add(obs, new_obs, action, reward, done_bool)

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
    
    print("Training done, about to save..")
    policy.save(filename='ddpg', directory=args.model_dir)
    print("Finished saving..should return now!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DQN Args
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true", default=True)  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=128, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.999, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--env_timesteps", default=500, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--replay_buffer_max_size", default=10000, type=int)  # Maximum number of steps to keep in the replay buffer
    parser.add_argument('--model-dir', type=str, default='reinforcement/pytorch/models/')
    parser.add_argument('--eps_start', type=float, default=0.9)
    parser.add_argument('--eps_end', type=float, default=0.05)
    parser.add_argument('--eps_decay', type=int, default=200)
    parser.add_argument('--target_update', type=int, default=10)

    _train(parser.parse_args())

    print("I'M DONE!!")