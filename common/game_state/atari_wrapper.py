#!/usr/bin/env python3
import gym
import logging
import pyglet
import cv2
import threading
import numpy as np

from time import sleep
from gym import spaces

logger = logging.getLogger("atari_wrapper")
__all__ = ['AtariWrapper']

class AtariWrapper(gym.Wrapper):
    """
    Sets the frame skip in ale and overrides step function
    in order to speed up game simulation
    Should only be used for Deterministic and NoFrameskip version of Atari gym
    """
    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)

        self.noop_max = noop_max

        # set frame skip in ALE
        self.unwrapped.ale.setInt('frame_skip'.encode('utf-8'), self.unwrapped.frameskip)
        self.unwrapped.seed()

        logger.info("lives: {}".format(self.unwrapped.ale.lives()))
        logger.info("frameskip: {} / {}".format(self.unwrapped.ale.getInt('frame_skip'.encode('utf-8')), self.unwrapped.frameskip))
        logger.info("repeat_action_probability: {}".format(self.unwrapped.ale.getFloat('repeat_action_probability'.encode('utf-8'))))
        logger.info("action_space: {}".format(self.env.action_space))

    def step(self, a):
        # frame skip is taken cared by ALE
        action = self.unwrapped._action_set[a]
        reward = self.unwrapped.ale.act(action)
        obs = self.unwrapped._get_obs()
        return obs, reward, self.unwrapped.ale.game_over(), {"ale.lives": self.unwrapped.ale.lives()}

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

        if self.env.unwrapped.frameskip == 4:
            skip = 3 if "SpaceInvaders" in self.env.spec.id else 4
            no_op = np.random.randint(1, self.noop_max * (skip//self.unwrapped.frameskip) + 1)
        else:
            no_op = 30

        assert no_op > 0
        for _ in range(no_op):
            obs, _, done, _ = self.env.step(0)
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
    def __init__(self, env, crop_screen=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.crop_screen = crop_screen
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        self.rgb_frame = frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if self.crop_screen:
            # crop
            gray_frame = gray_frame[34:34+160, :160]
            gray_frame = cv2.resize(gray_frame, (self.width, self.height),
                interpolation=cv2.INTER_AREA)
        else:
            # resize to height=84, width=84
            gray_frame = cv2.resize(gray_frame, (self.width, self.height),
                interpolation=cv2.INTER_AREA)

        return gray_frame


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0
