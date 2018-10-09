#!/usr/bin/env python3
"""
https://github.com/dbobrenko/reinforcement-learning-notes/blob/master/notes/dqn-agent.md
"""
import gym
import logging
import pyglet
import cv2
import threading
import numpy as np

from time import sleep
from gym import spaces

logger = logging.getLogger("atari_wrapper")

class AtariWrapper(gym.Wrapper):
    """
    Sets the frame skip in ale and overrides step function
    in order to speed up game simulation
    Should only be used for Deterministic and NoFrameskip version of Atari gym
    """
    def __init__(self, env, noop_max=30, skip=4):
        gym.Wrapper.__init__(self, env)

        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        self._skip = skip

        # set frame skip in ALE
        # self.unwrapped.ale.setInt('frame_skip'.encode('utf-8'), self.unwrapped.frameskip)
        # self.unwrapped.seed()

        logger.info("lives: {}".format(self.unwrapped.ale.lives()))
        logger.info("frameskip: {} / {}".format(self.unwrapped.ale.getInt('frame_skip'.encode('utf-8')), self.unwrapped.frameskip))
        logger.info("repeat_action_probability: {}".format(self.unwrapped.ale.getFloat('repeat_action_probability'.encode('utf-8'))))
        logger.info("action_space: {}".format(self.env.action_space))

    def step(self, a):
        return self.env.step(a)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

        skip = 3 if "SpaceInvaders" in self.env.spec.id else 4
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)

        assert noops > 0
        if self._skip == 1:
            for _ in range(noops):
                for i in range(skip):
                    obs, _, done, _ = self.env.step(self.noop_action)
                    if done:
                        break
                if done:
                    obs = self.env.reset(**kwargs)
        else:
            for _ in range(noops):
                obs, _, done, _ = self.env.step(self.noop_action)
                if done:
                    obs = self.env.reset(**kwargs)

        return obs

class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing."""
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

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
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
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

class HumanDemoEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        self.key = pyglet.window.key
        self.keys = self.key.KeyStateHandler()
        self.env.render(mode='human')
        self.env.unwrapped.viewer.window.push_handlers(self.keys)

        self.human_agent_action = 0
        self.human_agent_action_code = 0
        self.human_wants_restart = False
        self.human_sets_pause = False
        self.action_map = self.get_keys_to_action()
        self.stop_thread = False
        self.keys_thread = threading.Thread(target=(self.update_human_agent_action))
        self.keys_thread.start()

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, ac):
        return self.env.step(ac)

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'UP': self.key.UP,
            'DOWN': self.key.DOWN,
            'LEFT': self.key.LEFT,
            'RIGHT': self.key.RIGHT,
            'FIRE': self.key.SPACE,
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.unwrapped.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action

    def update_human_agent_action(self):
        while not self.stop_thread:
            key = []
            for k in [self.key.UP, self.key.DOWN, self.key.LEFT, self.key.RIGHT, self.key.SPACE]:
                if self.keys[k]:
                    key.append(k)
            key = tuple(sorted(key))
            action = self.action_map.get(key, 0)
            self.human_agent_action = action
            sleep(0.001)

    def close(self):
        self.stop_thread = True
        self.keys_thread.join()
        self.env.close()


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        self.rgb_frame = frame
        # convert to grayscale 210x160
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # resize to 110x84
        gray_frame = cv2.resize(gray_frame, (84, 110),
            interpolation=cv2.INTER_AREA)

        # crop 84x84
        gray_frame = gray_frame[18:110-8, :]

        return gray_frame


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


def get_wrapper_by_name(env, classname):
    """Given an a gym environment possibly wrapped multiple times, returns a wrapper
    of class named classname or raises ValueError if no such wrapper was applied
    """
    currentenv = env
    while True:
        if classname == currentenv.class_name():
            return currentenv
        elif isinstance(currentenv, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s" % classname)
