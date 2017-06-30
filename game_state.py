# -*- coding: utf-8 -*-
import sys
import numpy as np
import gym

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

class AtariEnvSkipping(gym.Wrapper):
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
        if self.env_id[:5] == 'Qbert':
            self.n_actions = 5
        elif self.env_id[:11] == 'NewBreakout':
            self.n_actions = 4
        print (colored("action space: {}".format(self.n_actions), "green"))

    def _step(self, a):
        if self.env_id[:5] == 'Qbert':
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
    def __init__(self, env_id=None, display=False, crop_screen=True, frame_skip=4, no_op_max=30, human_demo=False, auto_start=False):
        assert env_id is not None
        self._display = display
        self._crop_screen = crop_screen
        self._frame_skip = frame_skip
        if self._frame_skip < 1:
            self._frame_skip = 1
        self._no_op_max = no_op_max
        self.env_id = env_id
        self._human_demo = human_demo
        self._auto_start = auto_start

        env_id = self.env_id if self.env_id[:3] != 'New' else self.env_id[3:]
        self.env = gym.make(env_id)
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
            self.action_map[FIRE] = 1
            self.action_map[UP] = 2
            self.action_map[DOWN] = 3
            self.action_map[UPFIRE] = 4
            self.action_map[DOWNFIRE] = 5
        elif self.env_id[:5] == 'Qbert':
            self.action_map[UPRIGHT] = 1
            self.action_map[DOWNRIGHT] = 2
            self.action_map[UPLEFT] = 3
            self.action_map[DOWNLEFT] = 4
        elif self.env_id[:6] == 'Gopher':
            self.action_map[FIRE] = 1
            self.action_map[UP] = 2
            self.action_map[RIGHT] = 3
            self.action_map[LEFT] = 4
            self.action_map[UPFIRE] = 5
            self.action_map[RIGHTFIRE] = 6
            self.action_map[LEFTFIRE] = 7
        elif self.env_id[:7] == 'Freeway':
            self.action_map[UP] = 1
            self.action_map[DOWN] = 2
        elif self.env_id[:8] == 'Breakout':
            self.action_map[FIRE] = 1
            self.action_map[RIGHT] = 2
            self.action_map[LEFT] = 3
            self.action_map[RIGHTFIRE] = 4
            self.action_map[LEFTFIRE] = 5
        elif self.env_id[:9] == 'BeamRider':
            self.action_map[FIRE] = 1
            self.action_map[TORPEDO] = self.action_map[UP] = 2
            self.action_map[RIGHT] = 3
            self.action_map[LEFT] = 4
            self.action_map[RIGHTTORPEDO] = self.action_map[UPRIGHT] = 5
            self.action_map[LEFTTORPEDO] = self.action_map[UPLEFT] = 6
            self.action_map[RIGHTFIRE] = 7
            self.action_map[LEFTFIRE] = 8
        elif self.env_id[:13] == 'SpaceInvaders':
            self.action_map[FIRE] = 1
            self.action_map[RIGHT] = 2
            self.action_map[LEFT] = 3
            self.action_map[RIGHTFIRE] = 4
            self.action_map[LEFTFIRE] = 5
        elif self.env_id[:11] == 'NewBreakout':
            self.action_map[FIRE] = 1
            self.action_map[RIGHT] = 2
            self.action_map[LEFT] = 3
        else:
            # TODO: map ale action_set to right key actions
            pass

    def _init_keyboard(self):
        self.human_agent_action = 0
        self.human_agent_action_code = 0
        self.human_wants_restart = False
        self.human_sets_pause = False
        self.env.unwrapped._render(mode='human')
        self.key = pyglet.window.key
        self.keys = self.key.KeyStateHandler()
        self.env.unwrapped.env.viewer.window.push_handlers(self.keys)
        self.stop_thread = False
        self.keys_thread = threading.Thread(target=(self.update_human_agent_action))
        self.keys_thread.start()
        print ("Keys thread started")

    def close(self):
        self.stop_thread = True
        self.keys_thread.join()

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

    def _process_frame(self, action, normalize=True):
        info = {'lost_life': False}
        reward = 0
        observation, r, terminal, env_info = self.env.step(action)
        reward += r

        if self.lives < env_info['ale.lives']:
            self.lives = env_info['ale.lives']
        elif (self.lives - env_info['ale.lives']) != 0 and r <= 0:
            self.lives = env_info['ale.lives']
            info['lost_life'] = True

        if self._auto_start and info['lost_life']:
            if self.env_id[:8] == 'Breakout' or self.env_id[:11] == 'NewBreakout':
                observation, r, terminal, env_info = self.env.step(1)
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

        # normalize
        if normalize:
            x_t *= (1.0/255.0)
        return reward, terminal, x_t, info

    def reset(self, normalize=True):
        self.env.reset()
        self.lives = self.env.env.ale.lives()

        # randomize initial state
        if self._no_op_max > 0:
            no_op = np.random.randint(0, self._no_op_max * (4/self._frame_skip) + 1)
            for _ in range(no_op):
                self.env.step(0)

        _, _, x_t, info = self._process_frame(0, normalize=normalize)

        if self._auto_start:
            if self.env_id[:8] == 'Breakout' or self.env_id[:11] == 'NewBreakout':
                _, _, x_t, info = self._process_frame(1, normalize=normalize)

        self.x_t = x_t

        self.reward = 0
        self.terminal = False
        self.lost_life = info['lost_life']
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    def process(self, action, normalize=True):
        if self._display:
            self.env.unwrapped._render()

        if self._auto_start and not self._human_demo:
            if self.env_id[:8] == 'Breakout' or self.env_id[:11] == 'NewBreakout':
                action = action if action == 0 else action+1

        r, t, x_t1, info = self._process_frame(action, normalize=normalize)
        self.x_t = x_t1

        self.reward = r
        self.terminal = t
        self.lost_life = info['lost_life']
        x_t = np.reshape(x_t1, (84, 84, 1))
        self.s_t1 = np.append(self.s_t[:,:,1:], x_t, axis = 2)

    def update(self):
        self.s_t = self.s_t1

def test_keys(env_id):
    from skimage.measure import compare_ssim
    from skimage import io, filters
    test_game = GameState(env_id=env_id, display=True, frame_skip=1, human_demo=True, auto_start=True)
    test_game.reset()
    terminal = False
    skip = 0
    state = test_game.x_t
    while not test_game.terminal:
        a = test_game.human_agent_action
        test_game.process(a)
        new_state = test_game.x_t
        (score, diff) = compare_ssim(state, new_state, full=True)
        print("SSIM: {}".format(score))
        state = new_state
        edges = filters.sobel(state)
        cv2.imshow("edges", edges)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    test_game.env.close()
    test_game.close()
    del test_game.env
    del test_game

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    args = parser.parse_args()

    test_keys(args.env)
