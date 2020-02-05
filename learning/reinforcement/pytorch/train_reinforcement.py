import ast
import argparse
import logging
import torch

import os
import numpy as np

### Added test import
from reinforcement.pytorch.ddpg_per import DDPG_PER

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
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    # Set seeds
    seed(args.seed)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    episode_reward = None
    env_counter = 0
    reward = 0
    episode_timesteps = 0
    
    ### Added PER hyperparams
    prioritized_replay_alpha=0.6
    
    # Keep track of the best reward over time
    best_reward = -np.inf
    
    # Keep track of train_rewards
    train_rewards = []

    # Initialize policy
    if args.policy == 'ddpg':
        policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
        print("Initialized DDPG")
    if args.policy == 'td3':
        policy = TD3(state_dim, action_dim, max_action, net_type="cnn")
        print("Initialized TD3")
    
    ### Added per_ddpg
    if args.policy == 'ddpg_per':
        policy = DDPG_PER(state_dim, action_dim, max_action, net_type="cnn")
        print("Initialized DDPG with PER")
        
    # Evaluate untrained policy
    evaluations= [evaluate_policy(env, policy)]
    
    # Load previous policy
    if args.load_initial_policy:
    
        # Reset environment
        env_counter += 1
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
    
        args.start_timesteps=0
        
        checkpoint = torch.load('reinforcement/pytorch/models/' + args.policy)
        
        policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        evaluations = checkpoint['evaluations']
        total_timesteps = checkpoint['total_timesteps']
        train_rewards = checkpoint['train_rewards']
        episode_num = checkpoint['episode_num']
        
        if str(args.policy).lower() == 'ddpg':
            policy.critic.load_state_dict(checkpoint['critic_state_dict'])
        if str(args.policy).lower() == 'td3':
            policy.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
            policy.critic_2.load_state_dict(checkpoint['critic_1_state_dict'])
        
        
    ## Initialize ReplayBuffer
    if args.per:
        print('Training with Prioritized Experience Reply')
        replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_max_size, alpha = prioritized_replay_alpha)
    else:
        replay_buffer = ReplayBuffer(args.replay_buffer_max_size)

    print("Starting training")
    while total_timesteps < args.max_timesteps:

        #print("timestep: {} | reward: {}".format(total_timesteps, reward))

        if done:
            if total_timesteps != 0:
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                    total_timesteps, episode_num, episode_timesteps, episode_reward))
                policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)
                
                train_rewards.append(episode_reward)
                
                # Evaluate episode
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval %= args.eval_freq
                    eval_reward = evaluate_policy(env, policy)
                    evaluations.append(eval_reward)
                    print("\n--- rewards at time {}: {} ---".format(total_timesteps, eval_reward))

                    np.savetxt("reinforcement/pytorch/results/eval_rewards.csv", np.array(evaluations), delimiter=",")
                    np.savetxt("reinforcement/pytorch/results/train_rewards.csv", np.array(train_rewards), delimiter=",")

                    
                    # Save the policy according to the best reward over training
                    if eval_reward > best_reward:
                        best_reward = eval_reward
                        
                        save_dict = {
                        'total_timesteps': total_timesteps,
                        'evaluations': evaluations,
                        'train_rewards': train_rewards,
                        'episode_num': episode_num,
                        'actor_state_dict': policy.actor.state_dict()
                        }
                        
                        if str(args.policy).lower() == 'ddpg':
                            save_dict['critic_state_dict'] = policy.critic.state_dict()
                        if str(args.policy).lower() == 'td3':
                            save_dict['critic_1_state_dict'] = policy.critic_1.state_dict()
                            save_dict['critic_2_state_dict'] = policy.critic_2.state_dict()
                        
                        ### ADD ELSE ERROR
                        
                        print('Model saved\n')
                        torch.save(save_dict, args.model_dir + args.policy)
                        

            # Reset environment
            env_counter += 1
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1



        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.predict(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(
                    0,
                    args.expl_noise,
                    size=env.action_space.shape[0])
                          ).clip(env.action_space.low, env.action_space.high)

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

    print("Finished..should return now!")

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
