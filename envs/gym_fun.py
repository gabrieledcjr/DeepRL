#!/usr/bin/env python
import numpy
import gym
import pygame
import os
from pygame.locals import *
from sys import exit
import random
import cv2
from time import sleep
from termcolor import colored

position = 5, 325
os.environ['SDL_VIDEO_WINDOW_POS'] = str(position[0]) + "," + str(position[1])
pygame.init()
#screen = pygame.display.set_mode((640,480),pygame.NOFRAME)

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
            #self._screen = pygame.display.set_mode((240,320),0,32)

        sleep(2)
        self.reinit()

    def reinit(self, render=False, random_restart=False, terminate_loss_of_life=True):
        # self.loss_life = False
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

    def handle_user_event(self):
        pygame.event.get()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_KP8] or keys[pygame.K_UP]:
            action_index = 1
        elif keys[pygame.K_KP2] or keys[pygame.K_DOWN]:
            action_index = 2
        elif keys[pygame.K_KP4] or keys[pygame.K_LEFT]:
            action_index = 1
        elif keys[pygame.K_KP6] or keys[pygame.K_RIGHT]:
            action_index = 2
        elif keys[pygame.K_KP5]:
            action_index = 3
        else:
            action_index = 0

        return action_index

    def key_press(self, key, mod):
        if key==0xff0d: self.human_wants_restart = True
        if key==32: self.human_sets_pause = not self.human_sets_pause
        a = int( key - ord('0'))
        if (a > 0 and a <= 9) or (a >= 65409 and a <= 65417):
            self.human_agent_action_code = a
            if a == 8 or a == 65416: # UP
                self.human_agent_action = 1
            elif a == 2 or a == 65410: # DOWN
                self.human_agent_action = 2
            elif a == 4 or a == 65412: # LEFT
                self.human_agent_action = 1
            elif a == 6 or a == 65414: # RIGHT
                self.human_agent_action = 2
            elif a == 5 or a == 65413: # FIRE
                self.human_agent_action = 3
            else: # NOOP
                self.human_agent_action = 0

    def key_release(self, key, mod):
        a = int( key - ord('0') )
        if (a > 0 and a <= 9) or (a >= 65409 and a <= 65417):
            if self.human_agent_action_code == a:
                self.human_agent_action = 0

    def frame_step(self, act, render=False, random_restart=False):
        if self.game == 'pong':
            if act == 1:#Key up
                action = UP
            elif act == 2:#Key down
                action = DOWN
            else: # don't move
                action = 0
        elif self.game == 'breakout':
            if act == 1:#Key left
                action = LEFT
            elif act == 2:#Key right
                action = RIGHT
            elif act == 3: #FIRE
                action = FIRE
            else: # don't move
                action = 0
            # if self._human_demo and self.loss_life:
            #     action = FIRE # fire automatically just during HUMAN DEMO
            #     self.loss_life = False
        elif self.game == 'freeway':
            if act == 1:
                action = 1 # UP
            elif act == 2:
                action = 2 # DOWN
            else:
                action = 0

        observation, reward, terminal, info = self._env.step(action)
        self.screen_buffer = observation

        if (self.lives - info['ale.lives']) != 0:
            # self.loss_life = True
            self.lives -= 1
            # Consider terminal state after LOSS OF LIFE not after episode ends
            if self.terminate_loss_of_life:
                terminal = True

        # if self._human_demo:
        #     surface = pygame.surfarray.make_surface(observation)
        #     surface = pygame.transform.flip(surface, False, True)
        #     surface = pygame.transform.rotate(surface, -90)
        #     surface = pygame.transform.scale(surface, (240,320))
        #     bv = self._screen.blit(surface, (0,0))
        #     pygame.display.flip()
        if render or self._human_demo:
            self._env.render(mode='human')

        if terminal:
            self.reinit(random_restart=random_restart)

        return observation, reward, (1 if terminal else 0)

# test_game = GameState(human_demo=True, game='freeway')
# test_game.reinit(render=True)
# terminal = False
# skip = 0
# for t in range(2000):
#     a = test_game.human_agent_action
#     # if not skip:
#     #     a = test_game.human_agent_action
#     #     skip = 0
#     # else:
#     #     skip -= 1
#     _, r, terminal = test_game.frame_step(a, render=True)
#     sleep(0.06)
#     if terminal: break
#     if test_game.human_wants_restart: break
#     while test_game.human_sets_pause:
#         test_game._env.render(mode='human')
#         sleep(0.1)
