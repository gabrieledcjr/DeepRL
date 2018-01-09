# -*- coding: utf-8 -*-
import numpy as np
import time
import random
import sqlite3
import os
import logging

from datetime import datetime
from util import prepare_dir, process_frame, get_action_index, make_gif
from data_set import DataSet
from game_state import GameState

logger = logging.getLogger("a3c")

class CollectDemonstration(object):

    def __init__(
        self, game_state, resized_height, resized_width, phi_length, name,
        replay_memory=None, folder='', create_gif=False):
        """ Initialize collection of demo """
        assert folder != ''
        self.game_state = game_state
        self.resized_h = resized_height
        self.resized_w = resized_width
        self.phi_length = phi_length
        self.name = name
        self.D = replay_memory
        self.main_folder = folder

        # Create or connect to database
        self.conn = sqlite3.connect(
            self.main_folder + '/demo.db',
            detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
        self.db = self.conn.cursor()
        self._create_table()

        self._skip = 1
        if self.game_state.env.unwrapped.frameskip == 1:
            self._skip = 4
            if self.game_state.env_id[:13] == 'SpaceInvaders':
                self._skip = 3 # NIPS (makes laser always visible)

        self.create_gif = create_gif

    def _create_table(self):
        # Create table if doesn't exist
        self.db.execute(
            '''CREATE TABLE
               IF NOT EXISTS demo_samples
               (id INTEGER PRIMARY KEY,
                episode_num INTEGER,
                env_id TEXT,
                total_reward REAL,
                memory_size INTEGER,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration TEXT,
                total_steps INTEGER)''')
        self.db.execute(
            '''CREATE UNIQUE INDEX
               IF NOT EXISTS demo_samples_idx
               ON demo_samples(episode_num)''')

    def insert_data_to_db(self, demos=None):
        assert demos is not None
        self.db.executemany(
            '''INSERT OR REPLACE INTO
               demo_samples(episode_num, env_id, total_reward, memory_size, start_time, end_time, duration, total_steps)
               VALUES(?,?,?,?,?,?,?,?)''', demos)
        self.conn.commit()

    def _reset(self):
        self.game_state.reset(normalize=False)
        for _ in range(self.phi_length-1):
            self.D.add_sample(
                self.game_state.x_t,
                0,
                self.game_state.reward,
                self.game_state.terminal,
                self.game_state.lives,
                self.game_state.loss_life,
                self.game_state.gain_life)

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

            total_reward, total_steps, start_time, end_time, duration, mem_size = self.run(minutes_limit=minutes_limit, ep_num=ep, num_episodes=(num_episodes+start_ep)-1, demo_type=demo_type, model_net=model_net, D=D)
            rewards.append(total_reward)
            steps.append(total_steps)
            durations.append(duration)
            mem_sizes.append(mem_size)
            del D

            self.insert_data_to_db([(ep, self.name, total_reward, mem_size, start_time, end_time, str(duration), total_steps)])

        if demo_type == 0: # HUMAN
            self.game_state.stop_thread = True

        logger.debug("steps / episode: {}".format(steps))
        logger.debug("reward / episode: {}".format(rewards))
        logger.debug("mean steps: {} / mean reward: {}".format(np.mean(steps), np.mean(rewards)))
        total_duration = durations[0]
        logger.debug("duration / episode:")
        logger.debug("    {}".format(durations[0]))
        for j in range(1, len(durations)):
            logger.debug("    {}".format(durations[j]))
            total_duration += durations[j]
        logger.debug("total duration: {}".format(total_duration))
        logger.debug("mem size / episode: {}".format(mem_sizes))
        logger.debug("total memory size: {}".format(np.sum(mem_sizes)))
        logger.debug("total # of episodes: {}".format(num_episodes))
        self.conn.close()

    # def _pause_lost_life(self, is_breakout=False, is_beamrider=False):
    #     start_pause = datetime.now()
    #     pause_start = time.time()
    #     if is_breakout:
    #         key_str = '[FIRE]'
    #     elif is_beamrider:
    #         key_str = '[LEFT or RIGHT]'
    #     print ("You are required to press {} key to continue...".format(key_str))
    #     while True:
    #         action = self.game_state.human_agent_action
    #         if is_breakout and action == self.game_state.action_map[FIRE]:
    #             break
    #         if is_beamrider and action == self.game_state.action_map[LEFT]:
    #             break
    #         if is_beamrider and action == self.game_state.action_map[RIGHT]:
    #             break
    #         self.game_state.process(0, normalize=False)
    #     pause_duration = time.time() - pause_start
    #     logger.debug("Paused for {}".format(datetime.now() - start_pause))
    #     return action, pause_duration

    def run(self, minutes_limit=5, ep_num=0, num_episodes=0, demo_type=0, model_net=None, D=None):
        if D is not None:
            self.D = D

        if self.create_gif:
            gif_images = []

        rewards = {'train':[], 'eval':[]}

        full_episode = False
        if minutes_limit < 0:
            minutes_limit = 0
            full_episode = True
        timeout = 60 * minutes_limit
        t = 0
        is_reset = True
        total_reward = 0.0
        score1 = score2 = 0

        # re-initialize game for evaluation
        self._reset()

        rew = self.game_state.reward
        terminal = False
        lives = self.game_state.lives
        loss_life = self.game_state.loss_life
        gain_life = self.game_state.gain_life and not loss_life

        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()

        messagebox.showinfo(self.name, "Start episode {} of {}. Press OK to start playing".format(ep_num, num_episodes))

        # regular game
        start_time = datetime.now()
        timeout_start = time.time()

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
                    action = self.game_state.human_agent_action

            # store the transition in D
            self.D.add_sample(
                self.game_state.x_t,
                action,
                rew,
                self.game_state.terminal,
                lives,
                loss_life,
                gain_life)

            if self.create_gif:
                gif_images.append(self.game_state.x_t_rgb)

            # Ensure that D does not reach max memory that mitigate
            # problems when combining different human demo files
            if (self.D.size + 3) == self.D.max_steps:
                logger.warn("Memory max limit reached!")
                terminal = True

            if terminal:
                break

            rew = 0
            loss_life = False
            gain_life = False
            # when using frameskip=1, should repeat action four times
            for _ in range(self._skip):
                self.game_state.process(action, normalize=False)
                rew += self.game_state.reward
                lives = self.game_state.lives
                loss_life = loss_life or self.game_state.loss_life
                gain_life = (gain_life or self.game_state.gain_life) and not loss_life
                if not full_episode:
                    terminal = True if self.game_state.terminal or (time.time() > timeout_start + timeout) else False
                else:
                    terminal = self.game_state.terminal
                if terminal: break

            total_reward += rew
            t += 1
            self.game_state.update()

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info("Duration: {}".format(duration))
        logger.info("Total steps: {}".format(t))
        logger.info("Total reward: {}".format(total_reward))
        logger.info("Total Replay memory saved: {}".format(self.D.size))

        D.save(name=self.name, folder=self.folder, resize=True)
        if self.create_gif:
            time_per_step = 0.05
            make_gif(
                gif_images, self.folder+"demo.gif",
                duration=len(gif_images)*time_per_step,
                true_image=True, salience=False)

        return total_reward, t, start_time, end_time, duration, self.D.size

def get_demo(args):
    """
    Requirements: sudo apt-get install python3-tk
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --collect-demo --num-episodes=5 --demo-time-limit=5
    """
    if args.demo_memory_folder is not None:
        demo_memory_folder = 'demo_samples/{}'.format(args.demo_memory_folder)
    else:
        demo_memory_folder = 'demo_samples/{}'.format(args.gym_env.replace('-', '_'))

    if args.append_experiment_num is not None:
        demo_memory_folder += '_' + args.append_experiment_num

    prepare_dir(demo_memory_folder, empty=True)
    from log_formatter import LogFormatter
    fh = logging.FileHandler('{}/collect.log'.format(demo_memory_folder), mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = LogFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    game_state = GameState(env_id=args.gym_env, display=True, human_demo=True)
    collect_demo = CollectDemonstration(
        game_state,
        84, 84, 4,
        args.gym_env,
        replay_memory=None,
        folder=demo_memory_folder,
        create_gif=args.create_gif)
    collect_demo.run_episodes(
        args.num_episodes,
        minutes_limit=args.demo_time_limit,
        demo_type=0)

def test_collect(env_id):
    game_state = GameState(env_id=env_id, display=True, human_demo=True)
    test_folder = "demo_samples/{}_test".format(env_id.replace('-', '_'))
    prepare_dir(test_folder, empty=True)
    collect_demo = CollectDemonstration(
        game_state,
        84, 84, 4,
        env_id,
        replay_memory=None,
        folder=test_folder, create_gif=True)
    num_episodes = 1
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
