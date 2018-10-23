#!/usr/bin/env python3
import numpy as np
import time
import random
import sqlite3
import os
import coloredlogs, logging
import cv2

from collections import deque
from datetime import datetime
from common.util import prepare_dir, get_action_index, make_movie
from common.replay_memory import ReplayMemory
from common.game_state.atari_wrapper import get_wrapper_by_name

logger = logging.getLogger("collect_demo")

class CollectDemonstration(object):

    def __init__(
        self, game_state, resized_height, resized_width, phi_length, name,
        folder='', create_movie=False):
        """ Initialize collection of demo """
        assert folder != ''
        self.game_state = game_state
        self.resized_h = resized_height
        self.resized_w = resized_width
        self.phi_length = phi_length
        self.name = name
        self.main_folder = folder

        # Create or connect to database
        self.conn = sqlite3.connect(
            self.main_folder + '/demo.db',
            detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
        self.db = self.conn.cursor()
        self._create_table()

        self._skip = 4
        if "SpaceInvaders" in self.game_state.env.spec.id:
            self._skip = 3 # NIPS (makes laser always visible)

        self.create_movie = create_movie
        self.obs_buffer = np.zeros((2, 84 , 84), dtype=np.uint8)

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

    def _reset(self, replay_memory, hard_reset=True):
        self.game_state.reset(hard_reset=hard_reset)

        self.obs_buffer[0] = self.game_state.prev_x_t
        self.obs_buffer[1] = self.game_state.x_t
        max_obs = self.obs_buffer.max(axis=0)
        for _ in range(self.phi_length):
            replay_memory.add(
                max_obs,
                0,
                self.game_state.reward,
                self.game_state.terminal,
                self.game_state.lives,
                fullstate=self.game_state.fullstate)

    def _update_state_input(self, observation):
        self.state_input = np.roll(self.state_input, -1, axis=3)
        self.state_input[0, :, :, -1] = observation

    def run_episodes(self, num_episodes, start_ep=1, minutes_limit=5, demo_type=0, model_net=None):
        rewards = []
        steps = []
        durations = []
        mem_sizes = []
        for ep in range(start_ep, num_episodes+start_ep):
            replay_memory = ReplayMemory(
                self.resized_w, self.resized_h,
                np.random.RandomState(),
                max_steps=100000,
                phi_length=self.phi_length,
                num_actions=self.game_state.env.action_space.n,
                wrap_memory=False,
                full_state_size=self.game_state.clone_full_state().shape[0],
                clip_reward=True)

            self.folder = self.main_folder + '/{n:03d}/'.format(n=(ep))
            prepare_dir(self.folder, empty=True)

            total_reward, total_steps, start_time, end_time, duration, mem_size = self.run(
                minutes_limit=minutes_limit, ep_num=ep, num_episodes=(num_episodes+start_ep)-1,
                demo_type=demo_type, model_net=model_net, replay_memory=replay_memory)

            rewards.append(total_reward)
            steps.append(total_steps)
            durations.append(duration)
            mem_sizes.append(mem_size)
            del replay_memory

            self.insert_data_to_db([(ep, self.name, total_reward, mem_size, start_time, end_time, str(duration), total_steps)])

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
        self.game_state.close()
        self.conn.close()

    def run(self, minutes_limit=5, ep_num=0, num_episodes=0, demo_type=0, model_net=None, replay_memory=None):
        if self.create_movie:
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
        self._reset(replay_memory, hard_reset=True)

        rew = self.game_state.reward
        terminal = False
        lives = self.game_state.lives
        # loss_life = self.game_state.loss_life
        # gain_life = self.game_state.gain_life and not loss_life

        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()

        messagebox.showinfo(self.name, "Start episode {} of {}. Press OK to start playing".format(ep_num, num_episodes))

        # regular game
        start_time = datetime.now()
        timeout_start = time.time()

        actions = deque()

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
                    action = self.game_state.env.human_agent_action

            actions.append(action)
            self.game_state.step(action)
            rew += self.game_state.reward
            lives = self.game_state.lives
            # loss_life = loss_life or self.game_state.loss_life
            # gain_life = (gain_life or self.game_state.gain_life) and not loss_life
            total_reward += self.game_state.reward
            t += 1

            if self.create_movie:
                gif_images.append(self.game_state.get_screen_rgb())

            # Ensure that D does not reach max memory that mitigate
            # problems when combining different human demo files
            if (replay_memory.size + 3) == replay_memory.max_steps:
                logger.warn("Memory max limit reached!")
                terminal = True
            elif not full_episode:
                terminal = True if (time.time() > timeout_start + timeout) else False

            # add memory every 4th frame even if demo uses skip=1
            if self.game_state.get_episode_frame_number() % self._skip == 0 or terminal or self.game_state.terminal:
                self.obs_buffer[0] = self.game_state.x_t
                self.obs_buffer[1] = self.game_state.x_t1
                max_obs = self.obs_buffer.max(axis=0)
                # cv2.imshow('max obs', max_obs)
                # cv2.imshow('current', self.game_state.x_t1)
                # cv2.waitKey(1)

                # store the transition in D
                replay_memory.add(
                    max_obs,
                    actions.popleft(),
                    rew,
                    terminal or self.game_state.terminal,
                    lives,
                    fullstate=self.game_state.full_state1)
                actions.clear()
                rew = 0

                if terminal or get_wrapper_by_name(self.game_state.env, 'EpisodicLifeEnv').was_real_done:
                    break

                if self.game_state.terminal:
                    self._reset(replay_memory, hard_reset=False)
                    continue

            self.game_state.update()
            time.sleep(0.0167) #60 hz

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info("Duration: {}".format(duration))
        logger.info("Total steps: {}".format(t))
        logger.info("Total reward: {}".format(total_reward))
        logger.info("Total Replay memory saved: {}".format(replay_memory.size))

        replay_memory.save(name=self.name, folder=self.folder, resize=True)
        if self.create_movie:
            time_per_step = 0.0167
            make_movie(
                gif_images, self.folder+"demo",
                duration=len(gif_images)*time_per_step,
                true_image=True, salience=False)

        return total_reward, t, start_time, end_time, duration, replay_memory.size

def test_collect(env_id):
    from common.game_state import GameState
    game_state = GameState(env_id=env_id, display=True, human_demo=True)
    test_folder = "demo_samples/{}_test".format(env_id.replace('-', '_'))
    prepare_dir(test_folder, empty=True)
    collect_demo = CollectDemonstration(
        game_state,
        84, 84, 4,
        env_id,
        folder=test_folder, create_movie=True)
    num_episodes = 1
    collect_demo.run_episodes(
        num_episodes,
        minutes_limit=3,
        demo_type=0)

if __name__ == "__main__":
    coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    args = parser.parse_args()

    test_collect(args.env)
