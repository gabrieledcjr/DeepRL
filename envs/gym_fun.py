#!/usr/bin/env python
import argparse
import numpy
import gym
import os
from sys import exit
import random
import cv2
import pyglet
import threading
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

class GameState:
    def __init__(self, human_demo=False, frame_skip=4, game='pong'):
        self.game = game
        self._human_demo = human_demo

        if self._human_demo:
            self.action_map = {
                NOOP: 0, FIRE: 0, UP: 0, RIGHT: 0, LEFT: 0, DOWN: 0,
                UPRIGHT: 0, UPLEFT: 0, DOWNRIGHT: 0, DOWNLEFT: 0,
                UPFIRE: 0, RIGHTFIRE: 0, LEFTFIRE: 0, DOWNFIRE: 0,
                UPRIGHTFIRE: 0, UPLEFTFIRE: 0, DOWNRIGHTFIRE: 0, DOWNLEFTFIRE: 0,
                TORPEDO: 0, RIGHTTORPEDO: 0, LEFTTORPEDO: 0
            }
        if self.game == 'pong':
            self._env = gym.make('PongDeterministic-v3')
            self.n_actions = 6

            if self._human_demo:
                self.action_map[FIRE] = 1
                self.action_map[UP] = 2
                self.action_map[DOWN] = 3
                self.action_map[UPFIRE] = 4
                self.action_map[DOWNFIRE] = 5

        elif self.game == 'breakout':
            self._env = gym.make('BreakoutDeterministic-v3')
            self.n_actions = 4

            if self._human_demo:
                self.action_map[LEFT] = 1
                self.action_map[RIGHT] = 2
                self.action_map[FIRE] = 3

        elif self.game == 'freeway':
            self._env = gym.make('FreewayDeterministic-v3')
            self.n_actions = 3

            if self._human_demo:
                self.action_map[UP] = 1
                self.action_map[DOWN] = 2

        elif self.game == 'spaceinvaders':
            self._env = gym.make('SpaceInvadersDeterministic-v3')
            self.n_actions = 6

            if self._human_demo:
                self.action_map[FIRE] = 1
                self.action_map[RIGHT] = 2
                self.action_map[LEFT] = 3
                self.action_map[RIGHTFIRE] = 4
                self.action_map[LEFTFIRE] = 5

        elif self.game == 'qbert':
            self._env = gym.make('QbertDeterministic-v3')
            self.n_actions = 5

            if self._human_demo:
                self.action_map[UPRIGHT] = 1
                self.action_map[DOWNRIGHT] = 2
                self.action_map[UPLEFT] = 3
                self.action_map[DOWNLEFT] = 4

        elif self.game == 'beamrider':
            self._env = gym.make('BeamRiderDeterministic-v3')
            self.n_actions = 9

            if self._human_demo:
                self.action_map[FIRE] = 1
                self.action_map[TORPEDO] = 2
                self.action_map[RIGHT] = 3
                self.action_map[LEFT] = 4
                self.action_map[RIGHTTORPEDO] = 5
                self.action_map[LEFTTORPEDO] = 6
                self.action_map[RIGHTFIRE] = 7
                self.action_map[LEFTFIRE] = 8

        print (colored('{}Deterministic-v3'.format(self.game.title()), "green"))

        self._env.frameskip = frame_skip
        self.lives = self._env.ale.lives()
        print (colored("lives: {}".format(self.lives), "green"))
        print (colored("frameskip: {}".format(self._env.frameskip), "green"))
        print (colored("repeat_action_probability: {}".format(self._env.ale.getFloat(b'repeat_action_probability')), "green"))

        print (colored("human_demo: {}".format(self._human_demo), "green" if self._human_demo else "red"))

        if self._human_demo:
            self.human_agent_action = 0
            self.human_agent_action_code = 0
            self.human_wants_restart = False
            self.human_sets_pause = False
            self._env.render(mode='human')
            self.key = pyglet.window.key
            self.keys = self.key.KeyStateHandler()
            self._env.unwrapped.viewer.window.push_handlers(self.keys)
            self.stop_thread = False
            self.keys_thread = threading.Thread(target=(self.update_human_agent_action))
            self.keys_thread.start()

        sleep(2)
        self.reset()

    def reset(self, render=False, random_restart=False, terminate_loss_of_life=True):
        self.terminate_loss_of_life = terminate_loss_of_life
        if render or self._human_demo:
            self._env.render(mode='human')
        self._env.reset()
        self.lives = self._env.ale.lives()

        if random_restart:
            random_actions = random.randint(0, 30+1)
            for _ in range(random_actions):
                self.step(0)
        self.step(0)
        self.screen_buffer, _, _ = self.step(0)
        return self.screen_buffer

    def update_human_agent_action(self):
        while not self.stop_thread:
            action = NOOP
            if self.keys[self.key.DOWN] and self.keys[self.key.LEFT]:
                action = self.action_map[DOWNLEFT]
            elif self.keys[self.key.DOWN] and self.keys[self.key.RIGHT]:
                action = self.action_map[DOWNRIGHT]
            elif self.keys[self.key.DOWN] and self.keys[self.key.SPACE]:
                action = self.action_map[DOWNFIRE]
            elif self.keys[self.key.UP] and self.keys[self.key.LEFT]:
                action = self.action_map[UPLEFT]
            elif self.keys[self.key.UP] and self.keys[self.key.RIGHT]:
                action = self.action_map[UPRIGHT]
            elif self.keys[self.key.UP] and self.keys[self.key.SPACE]:
                action = self.action_map[UPFIRE]
            elif self.keys[self.key.LEFT] and self.keys[self.key.SPACE]:
                action = self.action_map[LEFTFIRE]
            elif self.keys[self.key.RIGHT] and self.keys[self.key.SPACE]:
                action = self.action_map[RIGHTFIRE]
            elif self.keys[self.key.LEFT] and self.keys[self.key.ENTER]:
                action = self.action_map[LEFTTORPEDO]
            elif self.keys[self.key.RIGHT] and self.keys[self.key.ENTER]:
                action = self.action_map[RIGHTTORPEDO]
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
            elif self.keys[self.key.ENTER]:
                action = self.action_map[TORPEDO]
            self.human_agent_action = action
            sleep(0.01)
        print ("Exited thread loop")

    def step(self, act, render=False, random_restart=False):
        # if self.game == 'pong':
        #     if act == 1: action = 2
        #     elif act == 2: action = 3
        #     else: action = NOOP
        if self.game == 'breakout':
            if act == 1: action = 3
            elif act == 2: action = 4
            elif act == 3: action = FIRE
            else: action = NOOP
        elif self.game == 'qbert':
            if act > 0: action = act + 1
            else: action = NOOP
        else:
            action = act

        observation, reward, terminal, info = self._env.step(action)
        self.screen_buffer = observation

        if (self.lives - info['ale.lives']) != 0:
            self.lives -= 1
            # Consider terminal state after LOSS OF LIFE not after episode ends
            if self.terminate_loss_of_life:
                terminal = True

        if render or self._human_demo:
            self._env.render(mode='human')

        if terminal:
            self.reset(random_restart=random_restart)

        return observation, reward, (1 if terminal else 0)

def test_game_1(env):
    test_game = GameState(human_demo=True, frame_skip=1, game=env)
    test_game.reset(render=True, terminate_loss_of_life=False)
    terminal = False
    skip = 0
    for t in range(5000):
        a = test_game.human_agent_action
        _, r, terminal = test_game.step(a, render=True)
        if terminal: break
        if test_game.human_wants_restart: break
        while test_game.human_sets_pause:
            test_game._env.render(mode='human')
            sleep(0.1)

def test_game_2(env):
    test_game = GameState(human_demo=True, frame_skip=1, game=env)
    test_game.reset(render=True)
    for t in range(10000):
        if t < 200:
            a = 8
        else:
            a = 2
        print ('action: ', a)
        _, r, terminal = test_game.step(a, render=True)
        if terminal: break

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    args = parser.parse_args()

    test_game_1(args.env)
    #test_game_2(args.env)
