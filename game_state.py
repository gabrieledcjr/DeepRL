# -*- coding: utf-8 -*-
import sys
import numpy as np
import gym
from gym.envs.atari.atari_env import ACTION_MEANING

import cv2
import pyglet
import threading
import coloredlogs, logging

from time import sleep
from termcolor import colored
from atari_wrapper import AtariWrapper

logger = logging.getLogger("a3c")

class GameState(object):
    def __init__(self, env_id=None, display=False, crop_screen=True, no_op_max=30, human_demo=False):
        assert env_id is not None
        self._display = display
        self._crop_screen = crop_screen
        self._no_op_max = no_op_max
        self.env_id = env_id
        self._human_demo = human_demo

        env = gym.make(self.env_id)
        if "Deterministic" in self.env_id or "NoFrameskip" in self.env_id:
            # necessary for faster simulation and to override keyboard controls
            self.env = AtariWrapper(env)
        else:
            self.env = env

        if self._human_demo:
            self._display = True
            self._init_keyboard()

        self.reset()

    def _init_keyboard(self):
        self.human_agent_action = 0
        self.human_agent_action_code = 0
        self.human_wants_restart = False
        self.human_sets_pause = False
        self.env.render(mode='human')
        self.key = pyglet.window.key
        self.keys = self.key.KeyStateHandler()
        self.action_map = self.env.get_keys_to_action(self.key)
        logger.info(self.env.unwrapped.get_action_meanings())
        self.env.unwrapped.viewer.window.push_handlers(self.keys)
        self.env.render(mode='human')
        self.stop_thread = False
        self.keys_thread = threading.Thread(target=(self.update_human_agent_action))
        self.keys_thread.start()
        logger.info("Keys thread started")

    def close(self):
        self.stop_thread = True
        self.keys_thread.join()

    def update_human_agent_action(self):
        while not self.stop_thread:
            action = 0
            key = []
            for k in [self.key.UP, self.key.DOWN, self.key.LEFT, self.key.RIGHT, self.key.SPACE]:
                if self.keys[k]:
                    key.append(k)
            key = tuple(sorted(key))
            if key in self.action_map:
                action = self.action_map[key]
            self.human_agent_action = action
            sleep(0.001)
        logger.warn("Exited thread loop")

    def _process_frame(self, observation, normalize=True):
        self.x_t_rgb = observation
        grayscale_observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        if self._crop_screen:
            resized_observation = grayscale_observation.astype(np.float32)
            # crop to fit 84x84
            x_t = resized_observation[34:34+160, :160]
            x_t = cv2.resize(x_t, (84, 84))
        else:
            # resize to height=84, width=84
            resized_observation = cv2.resize(grayscale_observation, (84,84))
            x_t = resized_observation.astype(np.float32)

        # normalize
        if normalize:
            x_t *= (1.0/255.0)
        return x_t

    def reset(self, normalize=True):
        self.env.reset()
        self.lives = self.env.unwrapped.ale.lives()

        # randomize initial state
        if self._no_op_max > 0:
            skip = 4 if self.env_id[:13] != 'SpaceInvaders' else 3
            no_op = np.random.randint(0, self._no_op_max * (skip//self.env.unwrapped.frameskip) + 1)
            for _ in range(no_op):
                self.env.step(0)

        observation, _, _, info = self._step(0)
        x_t = self._process_frame(observation, normalize=normalize)
        self.x_t = x_t

        self.reward = 0
        self.terminal = False
        self.lives = info['lives']
        self.loss_life = info['loss_life']
        self.gain_life = info['gain_life']
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    def _step(self, action):
        info = {'loss_life': False, 'gain_life': False}
        reward = 0
        observation, r, terminal, env_info = self.env.step(action)
        reward += r
        info['lives'] = env_info['ale.lives']

        if self.lives < env_info['ale.lives']:
            info['gain_life'] = True
        elif (self.lives - env_info['ale.lives']) != 0:
            info['loss_life'] = True

        return observation, reward, terminal, info

    def process(self, action, normalize=True):
        if self._display:
            self.env.render()

        observation, reward, terminal, info = self._step(action)
        x_t1 = self._process_frame(observation, normalize=normalize)
        self.x_t = x_t1

        self.reward = reward
        self.terminal = terminal
        self.lives = info['lives']
        self.loss_life = info['loss_life']
        self.gain_life = info['gain_life']
        x_t = np.reshape(x_t1, (84, 84, 1))
        self.s_t1 = np.append(self.s_t[:,:,1:], x_t, axis=2)

    def update(self):
        self.s_t = self.s_t1

def test_keys(env_id):
    from skimage.measure import compare_ssim
    from skimage import io, filters
    test_game = GameState(env_id=env_id, display=True, human_demo=True)
    terminal = False
    skip = 0
    state = test_game.x_t
    while not test_game.terminal:
        a = test_game.human_agent_action
        test_game.process(a)
        new_state = test_game.x_t
        (score, diff) = compare_ssim(state, new_state, full=True)
        # print("SSIM: {}".format(score))
        state = new_state
        # edges = filters.sobel(state)
        # cv2.imshow("edges", edges)
        # cv2.waitKey(1)
        print (a)
        if test_game.gain_life:
            print ("Gain Life")
        if test_game.loss_life:
            print ("Lost life!")
        if test_game.reward < 0:
            print (test_game.reward)

    # cv2.destroyAllWindows()
    test_game.env.close()
    test_game.close()
    del test_game.env
    del test_game

if __name__ == "__main__":
    coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s')
    from log_formatter import LogFormatter
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('test.log')
    fh.setLevel(logging.DEBUG)
    formatter = LogFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh.setFormatter(formatter)
    logger.addHandler(fh)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    args = parser.parse_args()

    test_keys(args.env)
