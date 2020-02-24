import ast
import argparse
import logging
import torch

import os
import numpy as np

### Added test import
#from reinforcement.pytorch.ddpg_per import DDPG_PER

# Duckietown Specific
from reinforcement.pytorch.ddpg import DDPG
from reinforcement.pytorch.td3 import TD3
from reinforcement.pytorch.utils import seed, evaluate_policy, ReplayBuffer, PrioritizedReplayBuffer
from utils.env import launch_env
from utils.wrappers import NormalizeWrapper, GrayscaleWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
from gym.wrappers import FrameStack

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Available policies
policies = {'ddpg': DDPG, 'td3': TD3}

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
    env = GrayscaleWrapper(env)
    env = NormalizeWrapper(env)
    env = FrameStack(env, 4)
    env = DtRewardWrapper(env)
    env = ActionWrapper(env)
    print("Initialized Wrappers")

    # Set seeds
    seed(args.seed)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Init training data
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    env_counter = 0
    reward = 0
    episode_timesteps = 0


    ### Added PER hyperparams
    # prioritized_replay_alpha=0.6

    # Keep track of the best reward over time
    best_reward = -np.inf

    # Keep track of train_rewards
    train_rewards = []
    
    # To print mean actions per episode
    mean_action = []

    # Initialize policy
    if args.policy not in policies:
        raise ValueError("Policy {} is not available, chose one of : {}".format(args.policy, list(policies.keys())))

    policy = policies[args.policy](state_dim, action_dim, max_action)

    # ### Added per_ddpg
    # if args.policy == 'ddpg_per':
    #     policy = DDPG_PER(state_dim, action_dim, max_action, net_type="cnn")
    #     print("Initialized DDPG with PER")

    # Evaluate untrained policy
    evaluations = [evaluate_policy(env, policy)]

    # Load previous policy
    if args.load_initial_policy:

        # Disable random start steps
        args.start_timesteps=0

        # Load training data
        checkpoint = load_training_state(args.model_dir, args.policy + "_training")

        evaluations = checkpoint['evaluations']
        total_timesteps = checkpoint['total_timesteps']
        train_rewards = checkpoint['train_rewards']
        episode_num = checkpoint['episode_num']
        best_reward = checkpoint['best_reward']

        # Load policy
        policy.load(args.model_dir, args.policy)

    ## Initialize ReplayBuffer
    if args.per:
        print('Training with Prioritized Experience Reply')
        replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_max_size, alpha = prioritized_replay_alpha)
    else:
        replay_buffer = ReplayBuffer(args.replay_buffer_max_size, args.batch_size, args.seed)

    print("Starting training")

    obs = env.reset()

    while total_timesteps < args.max_timesteps:

        # Select action
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.predict(np.array(obs))
            action = add_noise(action, args.expl_noise, env.action_space.low, env.action_space.high)
            
        mean_action.append(action)

        # Perform action
        new_obs, reward, done, _ = env.step(action)

        # Update episode reward
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add(obs, action, reward, new_obs, float(done))

        # Update network
        if len(replay_buffer) >= args.batch_size:
            policy.update(replay_buffer, args.discount, args.tau)

        # Update env
        obs = new_obs
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

        if episode_timesteps >= args.env_timesteps:
            done = True

        if done:
            print(("Total T: %d Episode Num: %d Episode T: %d Reward: %.2f Mean actions: %.4f , %.4f") % (
              total_timesteps, episode_num, episode_timesteps, episode_reward, np.mean(np.array(mean_action), axis=0)[0], np.mean(np.array(mean_action), axis=0)[1]))
                
            mean_action =  []

            train_rewards.append(episode_reward)

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                eval_reward = evaluate_policy(env, policy)
                evaluations.append(eval_reward)
                print("\n--- rewards at time {}: {} ---".format(total_timesteps, eval_reward))

                np.savetxt("reinforcement/pytorch/results/eval_rewards_" + args.policy + ".csv", np.array(evaluations), delimiter=",")
                np.savetxt("reinforcement/pytorch/results/train_rewards_" + args.policy + ".csv", np.array(train_rewards), delimiter=",")

                # Save the policy according to the best reward over training
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    policy.save(args.model_dir, args.policy)
                    save_training_state(args.model_dir, args.policy + "_training", best_reward, total_timesteps, evaluations, train_rewards, episode_num)

                    print('Model saved\n')

            # Reset environment
            obs = env.reset()
            env_counter += 1
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1


    print("Finished..should return now!")

def add_noise(action, expl_noise, low, high):
    if expl_noise != 0:
        action = (action + np.random.normal(
            0,
            expl_noise,
            size=action.shape)
            ).clip(low, high)
    return action


def save_training_state(directory, filename, best_reward, total_timesteps, evaluations, train_rewards, episode_num):
    save_dict = {
            'best_reward': best_reward,
            'total_timesteps': total_timesteps,
            'evaluations': evaluations,
            'train_rewards': train_rewards,
            'episode_num': episode_num
            }

    torch.save(save_dict, directory + filename)

def load_training_state(directory, filename):
    return torch.load(directory + filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # DDPG Args
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true", default=True)  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=32, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--env_timesteps", default=500, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--replay_buffer_max_size", default=10000, type=int)  # Maximum number of steps to keep in the replay buffer
    parser.add_argument('--model-dir', type=str, default='reinforcement/pytorch/models/')
    parser.add_argument('--load_initial_policy', help='Start the training on a loaded polisy', action = 'store_true')
    parser.add_argument('--policy', type=str, default='ddpg', help='Name of the initial policy')
    parser.add_argument('--per', help='Train with Prioritized Experience Replay', action = 'store_true')

    _train(parser.parse_args())
