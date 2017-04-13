# -*- coding: utf-8 -*-
import sys
import numpy as np
import gym

import cv2
import pyglet
import threading
#import atari_py

from time import sleep
from termcolor import colored

NOOP = 0
FIRE = 1
UP = 2
RIGHT = 3
LEFT = 4
DOWN = 5
UPRIGHT = 6
UPLEFT = 7
DOWNRIGHT = 8
DOWNLEFT = 9
UPFIRE = 10
RIGHTFIRE = 11
LEFTFIRE = 12
DOWNFIRE = 13
UPRIGHTFIRE = 14
UPLEFTFIRE = 15
DOWNRIGHTFIRE = 16
DOWNLEFTFIRE = 17

TORPEDO = 18
RIGHTTORPEDO = 19
LEFTTORPEDO = 20

class AtariEnvSkipping(gym.Wrapper):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, env, env_id=None, frameskip=4):
        self.env = env
        self.env_id = env_id
        self.env.ale.setFloat('frame_skip'.encode('utf-8'), frameskip)
        self.env.ale.setFloat('repeat_action_probability'.encode('utf-8'), 0.0)
        self.env._seed()

        print (colored("lives: {}".format(self.env.ale.lives()), "green"))
        print (colored("frameskip: {}".format(self.env.ale.getFloat(b'frame_skip')), "green"))
        print (colored("repeat_action_probability: {}".format(self.env.ale.getFloat(b'repeat_action_probability')), "green"))

        self.n_actions = self.env.action_space.n
        if self.env_id[:4] == "Pong":
            self.n_actions = 3
        elif self.env_id[:8] == 'Breakout':
            self.n_actions = 4
        elif self.env_id[:5] == 'Qbert':
            self.n_actions = 5
        print (colored("action space={}".format(self.n_actions), "green"))

    def _step(self, a):
        if self.env_id[:4] == "Pong":
            if a == 1: act = 2
            elif a == 2: act = 3
            else: act = 0
        elif self.env_id[:8] == 'Breakout':
            if a == 1: act = 3
            elif a == 2: act = 4
            elif a == 3: act = 1
            else: act = 0
        elif self.env_id[:5] == 'Qbert':
            if a > 0: act = a + 1
            else: act = 0
        else:
            act = a

        reward = 0.0
        action = self.env._action_set[act]

        reward += self.env.ale.act(action)
        ob = self.env._get_obs()

        return ob, reward, self.env.ale.game_over(), {"ale.lives": self.env.ale.lives()}

class GameState(object):
    def __init__(self, env_id=None, display=False, crop_screen=True, frame_skip=4, no_op_max=30, human_demo=False):
        assert env_id is not None
        self._display = display
        self._crop_screen = crop_screen
        self._frame_skip = frame_skip
        if self._frame_skip < 1:
            self._frame_skip = 1
        self._no_op_max = no_op_max
        self.env_id = env_id
        self._human_demo = human_demo

        self.env = gym.make(self.env_id)
        self.env = AtariEnvSkipping(self.env, env_id=self.env_id, frameskip=self._frame_skip)

        if self._human_demo:
            self.action_map = {
                NOOP: 0, FIRE: 0, UP: 0, RIGHT: 0, LEFT: 0, DOWN: 0,
                UPRIGHT: 0, UPLEFT: 0, DOWNRIGHT: 0, DOWNLEFT: 0,
                UPFIRE: 0, RIGHTFIRE: 0, LEFTFIRE: 0, DOWNFIRE: 0,
                UPRIGHTFIRE: 0, UPLEFTFIRE: 0, DOWNRIGHTFIRE: 0, DOWNLEFTFIRE: 0,
                TORPEDO: 0, RIGHTTORPEDO: 0, LEFTTORPEDO: 0
            }
            self._display = True
            self._remap_actions()
            self._init_keyboard()

        self.reset()

    def _remap_actions(self):
        if self.env_id[:4] == "Pong":
            self.action_map[UP] = 1
            self.action_map[DOWN] = 2
        elif self.env_id[:5] == 'Qbert':
            self.action_map[UPRIGHT] = 1
            self.action_map[DOWNRIGHT] = 2
            self.action_map[UPLEFT] = 3
            self.action_map[DOWNLEFT] = 4
        elif self.env_id[:7] == 'Freeway':
            self.action_map[UP] = 1
            self.action_map[DOWN] = 2
        elif self.env_id[:8] == 'Breakout':
            self.action_map[LEFT] = 1
            self.action_map[RIGHT] = 2
            self.action_map[FIRE] = 3
        elif self.env_id[:9] == 'BeamRider':
            self.action_map[FIRE] = 1
            self.action_map[TORPEDO] = self.action_map[UP] = 2
            self.action_map[RIGHT] = 3
            self.action_map[LEFT] = 4
            self.action_map[RIGHTTORPEDO] = self.action_map[UPRIGHT] = 5
            self.action_map[LEFTTORPEDO] = self.action_map[UPLEFT] = 6
            self.action_map[RIGHTFIRE] = 7
            self.action_map[LEFTFIRE] = 8
        elif self.env.id[:13] == 'SpaceInvaders':
            self.action_map[FIRE] = 1
            self.action_map[RIGHT] = 2
            self.action_map[LEFT] = 3
            self.action_map[RIGHTFIRE] = 4
            self.action_map[LEFTFIRE] = 5
        else:
            # TODO: map ale action_set to right key actions
            pass

    def _init_keyboard(self):
        self.human_agent_action = 0
        self.human_agent_action_code = 0
        self.human_wants_restart = False
        self.human_sets_pause = False
        self.env.render()
        self.key = pyglet.window.key
        self.keys = self.key.KeyStateHandler()
        self.env.unwrapped.env.viewer.window.push_handlers(self.keys)
        self.stop_thread = False
        self.keys_thread = threading.Thread(target=(self.update_human_agent_action))
        self.keys_thread.start()

    def update_human_agent_action(self):
        while not self.stop_thread:
            action = NOOP
            if self.keys[self.key.DOWN] and self.keys[self.key.LEFT]:
                action = self.action_map[DOWNLEFT]
            elif self.keys[self.key.DOWN] and self.keys[self.key.RIGHT]:
                action = self.action_map[DOWNRIGHT]
            elif self.keys[self.key.UP] and self.keys[self.key.LEFT]:
                action = self.action_map[UPLEFT]
            elif self.keys[self.key.UP] and self.keys[self.key.RIGHT]:
                action = self.action_map[UPRIGHT]
            elif self.keys[self.key.LEFT] and self.keys[self.key.SPACE]:
                print ("LEFTFIRE", self.action_map[LEFTFIRE])
                action = self.action_map[LEFTFIRE]
            elif self.keys[self.key.RIGHT] and self.keys[self.key.SPACE]:
                print ("RIGHTFIRE", self.action_map[RIGHTFIRE])
                action = self.action_map[RIGHTFIRE]
            elif self.keys[self.key.UP] and self.keys[self.key.SPACE]:
                action = self.action_map[UPFIRE]
            elif self.keys[self.key.DOWN] and self.keys[self.key.SPACE]:
                action = self.action_map[DOWNFIRE]
            elif self.keys[self.key.LEFT] and self.keys[self.key.ENTER]: # Torpedo in Beamrider
                action = self.action_map[UPLEFT]
            elif self.keys[self.key.RIGHT] and self.keys[self.key.ENTER]: # Torpedo in Beamrider
                action = self.action_map[UPRIGHT]
            elif self.keys[self.key.LEFT]:
                action = self.action_map[LEFT]
            elif self.keys[self.key.RIGHT]:
                action = self.action_map[RIGHT]
            elif self.keys[self.key.UP]:
                action = self.action_map[UP]
            elif self.keys[self.key.DOWN]:
                action = self.action_map[DOWN]
            elif self.keys[self.key.SPACE]:
                action = self.action_map[FIRE]
            elif self.keys[self.key.ENTER]: # Torpedo in Beamrider
                action = self.action_map[UP]
            self.human_agent_action = action
            sleep(0.01)
        print ("Exited thread loop")

    def _process_frame(self, action, reshape):
        reward = 0
        observation, r, terminal, _ = self.env.step(action)
        reward += r

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

        if reshape:
            x_t = np.reshape(x_t, (84, 84, 1))

        # normalize
        x_t *= (1.0/255.0)
        return reward, terminal, x_t

    def reset(self):
        self.env.reset()

        # randomize initial state
        if self._no_op_max > 0:
            no_op = np.random.randint(0, self._no_op_max + 1)
            for _ in range(no_op):
                self.env.step(0)

        _, _, x_t = self._process_frame(0, False)

        self.reward = 0
        self.terminal = False
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    def process(self, action):
        if self._display:
            self.env.render()

        r, t, x_t1 = self._process_frame(action, True)

        self.reward = r
        self.terminal = t
        self.s_t1 = np.append(self.s_t[:,:,1:], x_t1, axis = 2)

    def update(self):
        self.s_t = self.s_t1

def test_keys(env_id):
    test_game = GameState(env_id=env_id, display=True, frame_skip=1, human_demo=True)
    test_game.reset()
    terminal = False
    skip = 0
    while not test_game.terminal:
        a = test_game.human_agent_action
        test_game.process(a)
    test.game.env.close()
    del test.game.env
    del test.game

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    args = parser.parse_args()

    test_keys(args.env)
