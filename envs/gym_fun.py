#!/usr/bin/env python
import numpy
import gym
import os
from sys import exit
import random
import cv2
from time import sleep
from termcolor import colored

NOOP = 0
FIRE = 1
UP = 2
RIGHT = 4
LEFT = 3
DOWN = 5
# ACTION_MEANING = {
#     0 : "NOOP",
#     1 : "FIRE",
#     2 : "UP",
#     3 : "RIGHT",
#     4 : "LEFT",
#     5 : "DOWN",
#     6 : "UPRIGHT",
#     7 : "UPLEFT",
#     8 : "DOWNRIGHT",
#     9 : "DOWNLEFT",
#     10 : "UPFIRE",
#     11 : "RIGHTFIRE",
#     12 : "LEFTFIRE",
#     13 : "DOWNFIRE",
#     14 : "UPRIGHTFIRE",
#     15 : "UPLEFTFIRE",
#     16 : "DOWNRIGHTFIRE",
#     17 : "DOWNLEFTFIRE",
# }

class GameState:
    def __init__(self, human_demo=False, frame_skip=4, game='pong'):
        self.game = game
        if self.game == 'pong':
            self._env = gym.make('PongDeterministic-v3')
            self.n_actions = 3
            print (colored("PongDeterministic-v3", "green"))
        elif self.game == 'breakout':
            self._env = gym.make('BreakoutDeterministic-v3')
            self.n_actions = 4
            print (colored("BreakoutDeterministic-v3", "green"))
        elif self.game == 'freeway':
            self._env = gym.make('FreewayDeterministic-v3')
            self.n_actions = 3
            print (colored("FreewayDeterministic-v3", "green"))
        elif self.game == 'spaceinvaders':
            self._env = gym.make('SpaceInvadersDeterministic-v3')
            self.n_actions = self._env.action_space.n
            print (colored("SpaceInvadersDeterministic-v3", "green"))
        elif self.game == 'qbert':
            self._env = gym.make('QbertDeterministic-v3')
            self.n_actions = 5
            print (colored("QbertDeterministic-v3", "green"))

        self._env.frameskip = frame_skip
        self.lives = self._env.ale.lives()
        print (colored("lives: {}".format(self.lives), "green"))
        print (colored("frameskip: {}".format(self._env.frameskip), "green"))
        print (colored("repeat_action_probability: {}".format(self._env.ale.getFloat(b'repeat_action_probability')), "green"))

        self._human_demo = human_demo
        print (colored("human_demo: {}".format(self._human_demo), "green" if self._human_demo else "red"))
        if self._human_demo:
            self.human_agent_action = 0
            self.human_agent_action_code = 0
            self.human_wants_restart = False
            self.human_sets_pause = False
            self._env.render(mode='human')
            self._env.unwrapped.viewer.window.on_key_press = self.key_press
            self._env.unwrapped.viewer.window.on_key_release = self.key_release

        sleep(2)
        self.reinit()

    def reinit(self, render=False, random_restart=False, terminate_loss_of_life=True):
        self.terminate_loss_of_life = terminate_loss_of_life
        if render or self._human_demo:
            self._env.render(mode='human')
        self._env.reset()
        self.lives = self._env.ale.lives()

        if random_restart:
            random_actions = random.randint(0, 30+1)
            for _ in range(random_actions):
                self.frame_step(0)
        self.frame_step(0)
        self.screen_buffer, _, _ = self.frame_step(0)

    def _game_group_keys_1(self, a):
        """
        Breakout
        actions: 4 == [NOOP, L, R, Fire]
        """
        action = NOOP
        if a == 4 or a == 65412: # LEFT
            action = 1
        elif a == 6 or a == 65414: # RIGHT
            action = 2
        elif a == 5 or a == 65413: # FIRE
            action = 3
        return action

    def _game_group_keys_2(self, a):
        """
        Pong, Freeway
        actions: 3 = [NOOP, Up, Down]
        """
        action = NOOP
        if a == 8 or a == 65416: # UP
            action = 1
        elif a == 2 or a == 65410: # DOWN
            action = 2
        return action

    def _game_group_keys_3(self, a):
        """
        Space Invaders
        actions: 6 == [NOOP, FIRE, R, L, R_FIRE, L_FIRE]
        """
        action = NOOP
        if a == 4 or a == 65412: # LEFT
            action = 3
        elif a == 7 or a == 65415: # LEFTFIRE
            action = 5
        elif a == 6 or a == 65414: # RIGHT
            action = 2
        elif a == 9 or a == 65417: # RIGHTFIRE
            action = 4
        elif a == 5 or a == 65413: # FIRE
            action = 1
        return action

    def _game_group_keys_4(self, a):
        """
        Qbert
        actions: 5 == [NOOP, NOOP, R_UP, R_DOWN, L_UP, L_DOWN]
        """
        action = NOOP
        if a == 1 or a == 65409: # LEFTDOWN
            action = 4
        elif a == 7 or a == 65415: # LEFTUP
            action = 3
        elif a == 3 or a == 65411: # RIGHTDOWN
            action = 2
        elif a == 9 or a == 65417: # RIGHTUP
            action = 1
        return action

    def key_press(self, key, mod):
        if key==0xff0d: self.human_wants_restart = True
        if key==32: self.human_sets_pause = not self.human_sets_pause
        a = int( key - ord('0'))
        if (a > 0 and a <= 9) or (a >= 65409 and a <= 65417):
            self.human_agent_action_code = a
            if self.game == 'breakout':
                self.human_agent_action = self._game_group_keys_1(a)
            elif self.game == 'pong' or self.game == 'freeway':
                self.human_agent_action = self._game_group_keys_2(a)
            elif self.game == 'spaceinvaders':
                self.human_agent_action = self._game_group_keys_3(a)
            elif self.game == 'qbert':
                self.human_agent_action = self._game_group_keys_4(a)

    def key_release(self, key, mod):
        a = int( key - ord('0') )
        if (a > 0 and a <= 9) or (a >= 65409 and a <= 65417):
            if self.human_agent_action_code == a:
                self.human_agent_action = 0

    def frame_step(self, act, render=False, random_restart=False):
        if self.game == 'pong':
            if act == 1:
                action = UP
            elif act == 2:
                action = DOWN
            else:
                action = NOOP
        elif self.game == 'breakout':
            if act == 1:
                action = LEFT
            elif act == 2:
                action = RIGHT
            elif act == 3:
                action = FIRE
            else:
                action = NOOP
        elif self.game == 'qbert':
            if act > 0:
                action = act + 1
            else:
                action = NOOP
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
            self.reinit(random_restart=random_restart)

        return observation, reward, (1 if terminal else 0)

# test_game = GameState(human_demo=True, frame_skip=1, game='spaceinvaders')
# test_game.reinit(render=True)
# for t in range(5000):
#     if t < 200:
#         a = 4
#     else:
#         a = 5
#     print ('action: ', a)
#     _, r, terminal = test_game.frame_step(a, render=True)
#     if terminal: break

# test_game = GameState(human_demo=True, frame_skip=1, game='pong')
# test_game.reinit(render=True)
# terminal = False
# skip = 0
# for t in range(5000):
#     a = test_game.human_agent_action
#     # if not skip:
#     #     a = test_game.human_agent_action
#     #     skip = 0
#     # else:
#     #     skip -= 1
#     _, r, terminal = test_game.frame_step(a, render=True)
#     if terminal: break
#     if test_game.human_wants_restart: break
#     while test_game.human_sets_pause:
#         test_game._env.render(mode='human')
#         sleep(0.1)
