import collections

import numpy as np
import os

os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym import spaces
import cv2

cv2.ocl.setUseOpenCL(False)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class ChannelWrapper(gym.ObservationWrapper):
    def __init__(self, env, dtype=np.float32):
        super(ChannelWrapper, self).__init__(env)
        self.dtype = dtype

        obs_space = env.observation_space

        # old_space_low = old_space.low.reshape([1, *old_shape])
        # new_space_low = np.vstack()
        self.observation_space = \
            gym.spaces.Box(
                obs_space.low.reshape([1, *obs_space.shape]),
                obs_space.high.reshape([1, *obs_space.shape]),
                dtype=dtype
            )

    def observation(self, observation):
        return observation.reshape([1, *observation.shape])


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space =\
            gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(old_shape[-1], old_shape[0], old_shape[1]),
                dtype=np.float32
            )

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype

        # old_space = env.observation_space
        # old_shape = env.observation_space.shape
        #
        # # old_space_low = old_space.low.reshape([1, *old_shape])
        # # new_space_low = np.vstack()
        # self.observation_space = \
        #     gym.spaces.Box(
        #         old_space.low.repeat(n_steps, axis=0).reshape([n_steps, *old_shape]),
        #         old_space.high.repeat(n_steps, axis=0).reshape([n_steps, *old_shape]),
        #         dtype=dtype
        #     )

        old_space = env.observation_space
        self.observation_space = \
            gym.spaces.Box(
                old_space.low.repeat(n_steps, axis=0),
                old_space.high.repeat(n_steps, axis=0),
                dtype=dtype
            )

    def reset(self):
        self.buffer = np.zeros(self.observation_space.shape, dtype=self.dtype)

        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation

        # buffer.shape : [n_step, *observation.shape]
        return self.buffer


class FireResetEnv(gym.Wrapper):
    '''
    some games as Pong require a user to press the FIRE button to start the game.
    '''
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)

        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)

        if done:
            self.env.reset()

        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs
