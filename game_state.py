# -*- coding: utf-8 -*-
import sys
import numpy as np
import gym

import coloredlogs, logging

from time import sleep
from termcolor import colored
from atari_wrapper import AtariWrapper, FireResetEnv, HumanDemoEnv, WarpFrame

logger = logging.getLogger("a3c")

class GameState(object):
    def __init__(self, env_id=None, display=False, crop_screen=True, no_op_max=30, human_demo=False):
        assert env_id is not None
        self.display = display or human_demo
        self.env_id = env_id
        self.human_demo = human_demo
        self.fire_reset = False

        env = gym.make(self.env_id)
        assert "Deterministic" in env.spec.id or "NoFrameskip" in env.spec.id

        # necessary for faster simulation
        self.env = AtariWrapper(env, noop_max=no_op_max)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            self.fire_reset = True
            self.env = FireResetEnv(self.env)
        self.env = WarpFrame(self.env, crop_screen=crop_screen)
        # override keyboard controls for human demo
        if self.human_demo:
            self.env = HumanDemoEnv(self.env)
            logger.info(self.env.unwrapped.get_action_meanings())

        self.reset()

    def reset(self):
        x_t = self.env.reset()
        self.x_t1 = x_t
        self.full_state1 = self.env.unwrapped.clone_full_state()
        self.lives = self.env.unwrapped.ale.lives()
        self.reward = 0
        self.terminal = False
        self.loss_life = False
        self.gain_life = False
        self.s_t1 = np.stack((x_t, x_t, x_t, x_t), axis = 2)
        self.update()

    # def _step(self, action):
    #     info = {'loss_life': False, 'gain_life': False}
    #     obs, reward, terminal, env_info = self.env.step(action)
    #
    #     info['lives'] = env_info['ale.lives']
    #
    #     if self.lives < env_info['ale.lives']:
    #         info['gain_life'] = True
    #     elif (self.lives - env_info['ale.lives']) != 0:
    #         info['loss_life'] = True
    #
    #     return obs, reward, terminal, info

    def process(self, action):
        if self.display:
            self.env.render()

        obs, reward, terminal, env_info = self.env.step(action)

        self.loss_life = False
        self.gain_life = False
        if self.lives < env_info['ale.lives']:
            self.gain_life = True
        elif (self.lives - env_info['ale.lives']) != 0:
            self.loss_life = True
        self.x_t1 = obs
        self.full_state1 = self.env.unwrapped.clone_full_state()

        self.reward = reward
        self.terminal = terminal
        self.lives = env_info['ale.lives']
        x_t = np.reshape(obs, (84, 84, 1))
        self.s_t1 = np.append(self.s_t[:,:,1:], x_t, axis=2)

    def update(self):
        self.x_t = self.x_t1
        self.full_state = self.full_state1
        self.s_t = self.s_t1

    def clone_full_state(self):
        return self.env.unwrapped.clone_full_state()

    def restore_full_state(self, state):
        self.env.unwrapped.restore_full_state(state)

    def get_episode_frame_number(self):
        return self.env.unwrapped.ale.getEpisodeFrameNumber()

    def get_screen_rgb(self):
        return self.env.unwrapped.ale.getScreenRGB()

    def close(self):
        self.env.close()

def test_keys(env_id):
    import cv2
    from skimage.measure import compare_ssim
    from skimage import io, filters
    from collections import deque
    test_game = GameState(env_id=env_id, display=True, human_demo=True)
    terminal = False
    skip = 0
    state = test_game.x_t
    sys_state = None
    sys_states = deque(maxlen=100)
    last_num_steps = 0
    last_num_ctr = 0
    max_repeat = 5
    while not test_game.terminal:
        sys_state = test_game.clone_full_state()
        sys_states.append((sys_state, test_game.get_episode_frame_number()))
        print("frame number: ", test_game.get_episode_frame_number())
        a = test_game.env.human_agent_action
        test_game.process(a)
        # new_state = test_game.x_t
        # (score, diff) = compare_ssim(state, new_state, full=True)
        # print("SSIM: {}".format(score))
        # state = new_state
        # edges = filters.sobel(state)
        # cv2.imshow("edges", test_game.x_t)
        # cv2.waitKey(1)
        if test_game.gain_life:
            print ("Gain Life")
        if test_game.loss_life:
            print ("Lost life!")
            restore = True
            last_num_ctr += 1
            if last_num_steps == 0:
                last_num_steps = len(sys_states)
                print('last_num_steps={}'.format(last_num_steps))
            elif last_num_steps > len(sys_states):
                print('last_num_ctr={}'.format(last_num_ctr))
                if last_num_ctr == max_repeat:
                    restore = False
            if restore:
                full_state, frame_num = sys_states.popleft()
                print("\trestore frame number: ", frame_num)
                test_game.restore_full_state(full_state)
            steps = 0
            sys_states.clear()
        if test_game.reward > 0:
            last_num_steps = 0
            last_num_ctr = 0
            sys_states.clear()
        elif test_game.reward < 0:
            print ("Reward: ", test_game.reward)
            restore = True
            last_num_ctr += 1
            if last_num_steps == 0:
                last_num_steps = len(sys_states)
                print('last_num_steps={}'.format(last_num_steps))
            elif last_num_steps > len(sys_states):
                print('last_num_ctr={}'.format(last_num_ctr))
                if last_num_ctr == max_repeat:
                    restore = False
            if restore:
                full_state, frame_num = sys_states.popleft()
                print("\trestore frame number: ", frame_num)
                test_game.restore_full_state(full_state)
            steps = 0
            sys_states.clear()
        sleep(.02 * test_game.env.unwrapped.frameskip)

    # cv2.destroyAllWindows()
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
