# -*- coding: utf-8 -*-
import numpy as np
import time
import random
import sqlite3
import os
from datetime import datetime
from termcolor import colored
from util import prepare_dir, process_frame, save_compressed_images, get_action_index
from data_set import DataSet
from game_state import GameState, FIRE, LEFT, RIGHT

try:
    import cPickle as pickle
except ImportError:
    import pickle


class CollectDemonstration(object):

    def __init__(
        self, game_state, resized_height, resized_width, phi_length, name,
        replay_memory=None, folder=''):
        """ Initialize collection of demo """
        assert folder != ''
        self.game_state = game_state
        self.resized_h = resized_height
        self.resized_w = resized_width
        self.phi_length = phi_length
        self.name = name
        self.D = replay_memory
        self.main_folder = folder

        prepare_dir(self.main_folder, empty=False)

        # Create or connect to database
        self.conn = sqlite3.connect(self.main_folder + '/demo.db')
        self.db = self.conn.cursor()
        self._create_table()

        self._skip = 1
        if self.game_state.env.env.frameskip == 1:
            self._skip = 4
            if self.game_state.env_id[:13] == 'SpaceInvaders':
                self._skip = 3 # NIPS (makes laser always visible)

    def _create_table(self):
        # Create table if doesn't exist
        self.db.execute(
            '''CREATE TABLE
               IF NOT EXISTS demo_samples
               (id INTEGER PRIMARY KEY,
                episode_num INTEGER,
                env_id TEXT,
                total_reward REAL,
                memory_size INTEGER)''')
        self.db.execute(
            '''CREATE UNIQUE INDEX
               IF NOT EXISTS demo_samples_idx
               ON demo_samples(episode_num)''')

    def insert_data_to_db(self, demos=None):
        assert demos is not None
        self.db.executemany(
            '''INSERT OR REPLACE INTO
               demo_samples(episode_num, env_id, total_reward, memory_size)
               VALUES(?,?,?,?)''', demos)
        self.conn.commit()

    def _reset(self):
        self.game_state.reset(normalize=False)
        for _ in range(self.phi_length-1):
            self.D.add_sample(
                self.game_state.x_t,
                0,
                self.game_state.reward,
                self.game_state.terminal)

    def _update_state_input(self, observation):
        self.state_input = np.roll(self.state_input, -1, axis=3)
        self.state_input[0, :, :, -1] = observation

    def run_episodes(self, num_episodes, start_ep=1, minutes_limit=5, demo_type=0, model_net=None):
        rewards = []
        steps = []
        durations = []
        mem_sizes = []
        for ep in range(start_ep, num_episodes+start_ep):
            D = DataSet(
                self.resized_w, self.resized_h,
                np.random.RandomState(),
                max_steps=100000,
                phi_length=self.phi_length,
                num_actions=self.game_state.env.action_space.n)

            self.folder = self.main_folder + '/{n:03d}/'.format(n=(ep))
            prepare_dir(self.folder, empty=True)

            total_reward, total_steps, duration, mem_size = self.run(minutes_limit=minutes_limit, demo_type=demo_type, model_net=model_net, D=D)
            rewards.append(total_reward)
            steps.append(total_steps)
            durations.append(duration)
            mem_sizes.append(mem_size)
            del D

            self.insert_data_to_db([(ep, self.name, total_reward, mem_size)])

        if demo_type == 0: # HUMAN
            self.game_state.stop_thread = True

        print ("steps / episode:", steps)
        print ("reward / episode:", rewards)
        print ("Mean steps: {} / Mean reward: {}".format(np.mean(steps), np.mean(rewards)))
        total_duration = durations[0]
        print ("duration / episode:")
        print ("  ", durations[0])
        for j in range(1, len(durations)):
            print ("  ", durations[j])
            total_duration += durations[j]
        print ("Total duration:", total_duration)
        print ("mem size / episode:", mem_sizes)
        print ("Total memory size: {}".format(np.sum(mem_sizes)))
        print ("Total # of episodes: {}".format(num_episodes))
        self.conn.close()

    def _pause_lost_life(self, is_breakout=False, is_beamrider=False):
        start_pause = datetime.now()
        pause_start = time.time()
        if is_breakout:
            key_str = '[FIRE]'
        elif is_beamrider:
            key_str = '[LEFT or RIGHT]'
        print ("You are required to press {} key to continue...".format(key_str))
        while True:
            action = self.game_state.human_agent_action
            if is_breakout and action == self.game_state.action_map[FIRE]:
                break
            if is_beamrider and action == self.game_state.action_map[LEFT]:
                break
            if is_beamrider and action == self.game_state.action_map[RIGHT]:
                break
            self.game_state.process(0, normalize=False)
        pause_duration = time.time() - pause_start
        print ("Paused for {}".format(datetime.now() - start_pause))
        return action, pause_duration

    def run(self, minutes_limit=5, demo_type=0, model_net=None, D=None):
        if D is not None:
            self.D = D

        imgs = []
        acts = []
        rews = []
        terms = []

        rewards = {'train':[], 'eval':[]}

        # regular game
        start_time = datetime.now()
        timeout_start = time.time()
        full_episode = False
        if minutes_limit < 0:
            minutes_limit = 0
            full_episode = True
        timeout = 60 * minutes_limit
        t = 0
        terminal = False
        is_reset = True
        total_reward = 0.0
        rew = 0
        score1 = score2 = 0

        # re-initialize game for evaluation
        self._reset()
        time.sleep(2)

        is_breakout = self.game_state.env_id[:8] == 'Breakout'
        is_beamrider = self.game_state.env_id[:9] == 'BeamRider'
        override_add_sample = False
        if demo_type == 0 and (is_breakout or is_beamrider):
            action, pause_duration = self._pause_lost_life(is_breakout=is_breakout, is_beamrider=is_beamrider)
            timeout += pause_duration
            override_add_sample = True

        while True:
            if not terminal:
                if demo_type == 1: # RANDOM AGENT
                    action = np.random.randint(self.game_state.n_actions)
                elif demo_type == 2: # MODEL AGENT
                    if sub_t % self._skip == 0:
                        self._update_state_input(self.game_state.s_t)
                        readout_t = model_net.evaluate(self.state_input)[0]
                        action = get_action_index(readout_t, is_random=False, n_actions=self.game_state.n_actions)
                else: # HUMAN
                    if self.game_state.lost_life and (is_breakout or is_beamrider):
                        # when lost of life and their is a pause
                        action, pause_duration = self._pause_lost_life(is_breakout=is_breakout, is_beamrider=is_beamrider)
                        timeout += pause_duration
                        override_add_sample = True
                    else:
                        action = self.game_state.human_agent_action

            # store the transition in D
            self.D.add_sample(
                self.game_state.x_t,
                action,
                rew,
                self.game_state.terminal)

            total_reward += rew
            t += 1

            # Ensure that D does not reach max memory that mitigate
            # problems when combining different human demo files
            if (self.D.size + 3) == self.D.max_steps:
                print ("INFO: Memory max limit reached!")
                terminal = True

            if terminal:
                break

            rew = 0
            # when using frameskip=1, should repeat action four times
            for _ in range(self._skip):
                self.game_state.process(action, normalize=False)
                rew += self.game_state.reward
                if not full_episode:
                    terminal = True if self.game_state.terminal or (time.time() > timeout_start + timeout) else False
                else:
                    terminal = self.game_state.terminal
                if terminal: break

            self.game_state.update()


        duration = datetime.now() - start_time
        print ("Duration: {}".format(duration))
        print ("Total steps:", t)
        print ("Total reward:", total_reward)
        print ("Total Replay memory saved: {}".format(self.D.size))

        # Resize replay memory to exact memory size
        self.D.resize()
        data = {'D.width':self.D.width,
                'D.height':self.D.height,
                'D.max_steps':self.D.max_steps,
                'D.phi_length':self.D.phi_length,
                'D.num_actions':self.D.num_actions,
                'D.actions':self.D.actions,
                'D.rewards':self.D.rewards,
                'D.terminal':self.D.terminal,
                'D.size':self.D.size}
        images = self.D.imgs
        pkl_file = '{name}-dqn.pkl'.format(name=self.name)
        h5_file = '{name}-dqn-images.h5'.format(name=self.name)
        pickle.dump(data, open(self.folder + pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)
        print (colored('Compressing and saving replay memory...', 'blue'))
        save_compressed_images(self.folder + h5_file, images)
        print (colored('Compressed and saved replay memory', 'green'))

        return total_reward, t, duration, self.D.size

def get_demo(args):
    """
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --collect-demo --num-episodes=5 --demo-time-limit=5
    """
    if args.demo_memory_folder is not None:
        demo_memory_folder = 'demo_samples/{}'.format(args.demo_memory_folder)
    else:
        demo_memory_folder = 'demo_samples/{}'.format(args.gym_env.replace('-', '_'))

    game_state = GameState(env_id=args.gym_env, display=True, human_demo=True)
    collect_demo = CollectDemonstration(
        game_state,
        84, 84, 4,
        args.gym_env,
        replay_memory=None,
        folder=demo_memory_folder)
    collect_demo.run_episodes(
        args.num_episodes,
        minutes_limit=args.demo_time_limit,
        demo_type=0)

def test_collect(env_id):
    game_state = GameState(env_id=env_id, display=True, human_demo=True)
    test_folder = env_id.replace('-', '_') + "_test_demo_samples"
    collect_demo = CollectDemonstration(
        game_state,
        84, 84, 4,
        env_id,
        replay_memory=None,
        folder=test_folder)
    num_episodes = 2
    collect_demo.run_episodes(
        num_episodes,
        minutes_limit=1,
        demo_type=0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    args = parser.parse_args()

    test_collect(args.env)
