import gym
from gym import spaces
import numpy as np
from gym_duckietown.simulator import NotInLane


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(80, 60, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):
        from PIL import Image
        return np.array(Image.fromarray(observation).resize(self.shape[0:2]))

class GrayscaleWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(GrayscaleWrapper, self).__init__(env)
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            self.observation_space.shape[:-1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        from PIL import Image
        return np.array(Image.fromarray(observation).convert('L'))

class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0]
        self.obs_hi = self.observation_space.high[0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)

class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        speed = self.env.speed
        try :
            lp = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
        except NotInLane:
            return -1

        reward_angle = np.clip(1 - lp.angle_rad**2, -1, 1)
        reward_dist = np.clip(1 - lp.dist**2, -1, 1)
        reward_speed = self.env.speed


        return (reward_angle + reward_dist + reward_speed) / 3


# Deprecated
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)
        self.prev_action = []

    def action(self, action):
        max_delta_steering = 0.1
        if self.prev_action != []:
            # limit
            prev_steering = self.prev_action[1]
            steering = np.clip(action[1], prev_steering - max_delta_steering, prev_steering + max_delta_steering)

            action = [action[0], steering]

        self.prev_action = action
        return action
