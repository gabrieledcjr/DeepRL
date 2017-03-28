#!/usr/bin/env python3
import argparse
from numpy.random import RandomState

import envs.gym_fun as game
from data_set import DataSet
from util import get_compressed_images, save_compressed_images

try:
    import cPickle as pickle
except ImportError:
    import pickle

def aggregate_demo(args):
    """
    python3 aggregate_demo.py pong --range-start=0 --range-end=5
    """

    if args.demo_memory_folder is not None:
        demo_memory_folder = args.demo_memory_folder
    else:
        demo_memory_folder = "{}_demo_samples".format(args.env)

    game_state = game.GameState(game=args.env)
    D = DataSet(
        args.resized_height, args.resized_width, RandomState(),
        args.replay_memory, args.phi_len, game_state.n_actions)

    data_file = '{}-dqn.pkl'.format(args.env)
    img_file = '{}-dqn-images.h5'.format(args.env)
    for index in range(args.range_start, args.range_end):
        print ("Demonstration sample #{num:03d}".format(num=index+1))
        try:
            data = pickle.load(open(demo_memory_folder + '/{0:03d}/'.format(index+1) + data_file, 'rb'))
        except:
            print ("Check demo folder if it exist!")
            return
        actions = data['D.actions']
        rewards = data['D.rewards']
        terminal = data['D.terminal']

        imgs = get_compressed_images(demo_memory_folder + '/{0:03d}/'.format(index+1) + img_file + '.gz')
        print ("\tMemory size: {}".format(data['D.size']))
        for mem_index in range(data['D.size']):
            D.add_sample(imgs[mem_index], actions[mem_index], rewards[mem_index], terminal[mem_index])
        # h5file.close()
        print ("\tTotal Memory size: {}".format(D.size))

    D.resize()
    D.create_validation_set(percent=args.validation_set_percent)

    data = {'D.width': D.width,
            'D.height': D.height,
            'D.max_steps': D.max_steps,
            'D.phi_length': D.phi_length,
            'D.num_actions': D.num_actions,
            'D.actions': D.actions,
            'D.rewards': D.rewards,
            'D.terminal': D.terminal,
            'D.bottom': D.bottom,
            'D.top': D.top,
            'D.size': D.size,
            'D.validation_set_markers': D.validation_set_markers,
            'D.validation_indices': D.validation_indices,
            'epsilon': args.init_epsilon,
            't': 0}
    images = D.imgs

    pickle.dump(data, open(demo_memory_folder + '/' + args.env + '-dqn-all.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    print ("Saving and compressing replay memory...")
    save_compressed_images(demo_memory_folder + '/' + args.env + '-dqn-images-all.h5', images)
    print ("Saved and compressed replay memory")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)

    parser.add_argument('--init-epsilon', type=float, default=1.0)
    parser.add_argument('--replay-memory', type=int, default=1000000)
    parser.add_argument('--resized-width', type=int, default=84)
    parser.add_argument('--resized-height', type=int, default=84)
    parser.add_argument('--phi-len', type=int, default=4)

    parser.add_argument('--range-start', type=int, default=0)
    parser.add_argument('--range-end', type=int, default=5)

    parser.add_argument('--validation-set-percent', type=float, default=0.2)

    parser.add_argument('--demo-memory-folder', type=str, default=None)

    args = parser.parse_args()
    assert args.range_start < args.range_end

    aggregate_demo(args)

if __name__ == "__main__":
    main()
