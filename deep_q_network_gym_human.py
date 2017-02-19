#!/usr/bin/env python
import cv2
import sys, getopt
import envs.gym_fun as game
import random
import numpy as np
import math
from data_set import DataSet
import tables
import time
from datetime import datetime
from util import prepare_dir

try:
    import cPickle as pickle
except ImportError:
    import pickle

# Breakout
GAME = 'breakout' # the name of the game being played for log files
ACTIONS = 4 # number of valid actions

# Pong
# GAME = 'pong' # the name of the game being played for log files
# ACTIONS = 3 # number of valid actions

GAMMA = 0.99 # decay rate of past observations
OBSERVE = 50000 # timesteps to observe before training
EXPLORE = 1000000 # frames over which to anneal epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 200000 # number of previous transitions to remember
RESIZED_WIDTH = 84
RESIZED_HEIGHT = 84
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others

PHI_LENGTH = 4 # rms

SAVE_FREQ = 50000
EVAL_FREQ = 5000
EVAL_MAX_STEPS = 50000
C_FREQ = 10000

def resetGame(game_state):
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    observation, r_0, terminal = game_state.frame_step(0)

    observation = cv2.cvtColor(cv2.resize(observation, (RESIZED_HEIGHT, RESIZED_WIDTH)), cv2.COLOR_BGR2GRAY)
    observation_t = observation / 255.0
    s_t = [ observation_t for _ in range(PHI_LENGTH)]
    s_t = np.stack(tuple(s_t), axis = 2)
    return observation, s_t

MAX_STEPS = 50
IS_TESTING = True
def trainNetwork(D, file_num=''):
    # open up a game state to communicate with emulator
    game_state = game.GameState(human_demo=True, frame_skip=4, game=GAME)

    imgs = []
    acts = []
    rews = []
    terms = []

    rewards = {'train':[], 'eval':[]}

    # regular game
    start_time = datetime.now()
    timeout_start = time.time()
    timeout = 60 * 5 # 300 seconds
    t = 0
    terminal = False
    # re-initialize game for evaluation
    game_state.reinit(gui=False, random_restart=True, is_testing=IS_TESTING)
    last_img, s_t = resetGame(game_state)
    is_reset = True
    total_reward = 0.0
    score1 = score2 = 0
    sub_t = 0
    sub_r = 0
    rewards = []
    sub_steps = []
    total_episodes = 0
    while True:
        a_t = np.zeros([ACTIONS])
        action_index = 0
        action_index = game_state.handle_user_event()
        if GAME == 'breakout' and is_reset:
            action_index = 3 #FIRE
            is_reset = False

        a_t[action_index] = 1
        observation, r_t, terminal = game_state.frame_step(action_index, gui=False, random_restart=True)
        print ("reward: ", r_t)
        observation = cv2.cvtColor(cv2.resize(observation, (RESIZED_HEIGHT, RESIZED_WIDTH)), cv2.COLOR_BGR2GRAY)
        #terminal = True if terminal or r_t==-1 else False
        terminal = True if terminal or (time.time() > timeout_start + timeout) else False
        # store the transition in D
        D.add_sample(last_img, a_t, r_t, terminal)
        last_img = observation

        if r_t > 0:
            sub_r += r_t
            total_reward += r_t

        time.sleep(0.08)
        sub_t += 1
        t += 1

        if terminal:
            total_episodes += 1
            rewards.append(sub_r)
            sub_steps.append(sub_t)
            sub_r = 0
            sub_t = 0
            game_state.reinit(gui=False, random_restart=True, is_testing=IS_TESTING)
            last_img, s_t = resetGame(game_state)
            is_reset = True
            time.sleep(0.5)

            if time.time() > timeout_start + timeout:
                break

    print "Total episodes: {}".format(total_episodes)
    print "Steps per episode:", sub_steps
    print "Reward per episode:", rewards
    print "Total steps: {} / Total reward: {}".format(t,total_reward)
    print "Duration: {}".format(datetime.now() - start_time)
    print "Total Replay memory saved: {}".format(len(D))

    data = {'D.width':D.width,
            'D.height':D.height,
            'D.max_steps':D.max_steps,
            'D.phi_length':D.phi_length,
            'D.num_actions':D.num_actions,
            'D.actions':D.actions,
            'D.rewards':D.rewards,
            'D.terminal':D.terminal,
            'D.bottom':D.bottom,
            'D.top':D.top,
            'D.size':D.size}
    images = D.imgs
    pickle.dump(data, open('{}_human_samples/{}/'.format(GAME, file_num[1:]) + GAME + '-dqn' + file_num + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    h5file = tables.open_file('{}_human_samples/{}/'.format(GAME, file_num[1:]) + GAME + '-dqn-images' + file_num + '.h5', mode='w', title='Images Array')
    root = h5file.root
    h5file.create_array(root, "images", images)
    h5file.close()


def playGame(file_num):
    if True: # Deterministic
        rng = np.random.RandomState(123456)
    else:
        rng = np.random.RandomState()
    replay_memory = DataSet(RESIZED_WIDTH, RESIZED_HEIGHT, rng, REPLAY_MEMORY, PHI_LENGTH, ACTIONS)
    trainNetwork(replay_memory, file_num='-'+str(file_num))

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hn:",["file_num="])
    except getopt.GetoptError:
          print 'deep_q_network_pong_human_gym.py -n <file number>'
          sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'deep_q_network_pong_human_gym.py -n <file number>'
            sys.exit()
        elif opt in ("-n", "--file_num"):
            file_num = arg
    prepare_dir(GAME + '_human_samples/' + file_num, empty=True)
    playGame(file_num)

if __name__ == "__main__":
    main(sys.argv[1:])
