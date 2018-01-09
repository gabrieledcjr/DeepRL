# -*- coding: utf-8 -*-
import gym
import logging

logger = logging.getLogger("a3c")
__all__ = ['AtariWrapper']

class AtariWrapper(gym.Wrapper):
    """
        Sets the frame skip in ale and overrides step function
        in order to speed up game simulation
        Should only be used for Deterministic and NoFrameskip version of Atari gym
    """
    def __init__(self, env):
        super(AtariWrapper, self).__init__(env)

        self.unwrapped.ale.setInt('frame_skip'.encode('utf-8'), self.unwrapped.frameskip)
        self.unwrapped._seed()

        logger.info("lives: {}".format(self.unwrapped.ale.lives()))
        logger.info("frameskip: {} / {}".format(self.unwrapped.ale.getInt('frame_skip'.encode('utf-8')), self.unwrapped.frameskip))
        logger.info("repeat_action_probability: {}".format(self.unwrapped.ale.getFloat('repeat_action_probability'.encode('utf-8'))))
        logger.info("action_space: {}".format(self.env.action_space))

    def _step(self, a):
        action = self.unwrapped._action_set[a]
        reward = self.unwrapped.ale.act(action)
        ob = self.unwrapped._get_obs()
        return ob, reward, self.unwrapped.ale.game_over(), {"ale.lives": self.unwrapped.ale.lives()}

    def get_keys_to_action(self, keystroke):
        KEYWORD_TO_KEY = {
            'UP': keystroke.UP,
            'DOWN': keystroke.DOWN,
            'LEFT': keystroke.LEFT,
            'RIGHT': keystroke.RIGHT,
            'FIRE': keystroke.SPACE,
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
