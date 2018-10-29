#!/usr/bin/env python3
import re
import logging
import sys

from common.util import prepare_dir

try:
    import cPickle as pickle
except ImportError:
    import pickle

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def strip_ansi_codes(s):
    return re.sub(r'\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?(;[0-9]{3})?)?[m|K]?', '', s)

for log_num in [1]:
    rl_alg = "a3c"
    atari_game = "Breakout"
    path = "results_kamiak/slurm_log"
    log_file = "{}_{}_base_{}.log".format(rl_alg, atari_game, log_num)


    folder = "results_kamiak/{}/{}NoFrameskip_v4_mnih2015_lstm_{}".format(rl_alg, atari_game, log_num)
    rewards_file = "{}NoFrameskip_v4-{}-rewards.pkl".format(atari_game, rl_alg)

    prepare_dir(folder, empty=False)

    rewards = {'train':{}, 'eval':{}}
    with open(path + '/' + log_file, 'r') as fp:
        for line in fp:
            if re.search("final score", line):
                line = line.rstrip()
                line = line.split(" ")
                global_t = int(line[5].split("=")[-1])
                ave_score = float(line[8].split("=")[-1])
                ave_steps = int(line[10].split("=")[-1])
                num_episodes = int(line[12].split("=")[-1])
                logger.debug("test: global_t={0:9d} score={1:.2f} steps={2} episodes={3}".format(global_t, ave_score, ave_steps, num_episodes))
                rewards['eval'][global_t] = (ave_score, ave_steps, num_episodes)
            elif re.search("DEBUG train:", line):
                line = line.rstrip()
                line = line.split(" ")
                global_t = int(line[6].split("=")[-1])
                score = float(strip_ansi_codes(line[7]).split("=")[-1])
                steps = int(strip_ansi_codes(line[8]).split("=")[-1])
                logger.debug("train: global_t={} score={} steps={}".format(global_t, score, steps))
                rewards['train'][global_t] = (score, steps)

    pickle.dump(rewards, open(folder + '/' + rewards_file, 'wb'), pickle.HIGHEST_PROTOCOL)
    logger.info('Data saved at {}'.format(folder + '/' + rewards_file))