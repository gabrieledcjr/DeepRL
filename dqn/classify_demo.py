#!/usr/bin/env python3
import numpy as np
import sys
import os
import logging

from common.replay_memory import ReplayMemory
from common.game_state import GameState
from common.util import get_compressed_images

logger = logging.getLogger("classify_demo")

try:
    import cPickle as pickle
except ImportError:
    import pickle

class ClassifyDemo(object):
    def __init__(
        self, net, replay_memory, name, train_max_steps, batch_size,
        eval_freq, demo_memory_folder='', folder=''):
        """ Initialize Classifying Human Demo Training """
        self.net = net
        self.replay_memory = replay_memory
        self.name = name
        self.train_max_steps = train_max_steps
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.demo_memory_folder = demo_memory_folder
        self.folder = folder

        self._load_memory()

    def _load_memory(self):
        print ("Loading data")
        if self.name == 'pong' or self.name == 'breakout':
            # data were pickled using Python 2 which have compatibility issues in Python 3
            data = pickle.load(open('{}/{}-dqn-all.pkl'.format(self.demo_memory_folder, self.name), 'rb'), encoding='latin1')
        else:
            data = pickle.load(open('{}/{}-dqn-all.pkl'.format(self.demo_memory_folder, self.name), 'rb'))

        self.replay_memory.width = data['D.width']
        self.replay_memory.height = data['D.height']
        self.replay_memory.max_steps = data['D.max_steps']
        self.replay_memory.phi_length = data['D.phi_length']
        self.replay_memory.num_actions = data['D.num_actions']
        self.replay_memory.actions = data['D.actions']
        self.replay_memory.rewards = data['D.rewards']
        self.replay_memory.terminal = data['D.terminal']
        self.replay_memory.bottom = data['D.bottom']
        self.replay_memory.top = data['D.top']
        self.replay_memory.size = data['D.size']
        self.replay_memory.validation_set_markers = data['D.validation_set_markers']
        self.replay_memory.validation_indices = data['D.validation_indices']
        self.replay_memory.imgs = get_compressed_images('{}/{}-dqn-images-all.h5'.format(self.demo_memory_folder, self.name) + '.gz')
        print ("Data loaded!")

    def run(self):
        data = {
            'training_step': [],
            'training_accuracy': [],
            'training_entropy': [],
            'testing_step': [],
            'testing_accuracy': [],
            'testing_entropy': [],
            'max_accuracy': 0.,
            'max_accuracy_step': 0,
        }
        max_val = -(sys.maxsize),
        no_change_ctr = 0

        s_j_batch_validation, a_batch_validation = self.replay_memory.get_validation_set()
        for i in range(self.train_max_steps):
            s_j_batch, a_batch, _, _, _ = self.replay_memory.random_batch(self.batch_size, exclude_validation=True)

            if (i % self.eval_freq) == 0:
                entropy, acc, _, _ = self.net.evaluate_batch(s_j_batch_validation, a_batch_validation)
                data['testing_step'].append(i)
                data['testing_accuracy'].append(acc)
                data['testing_entropy'].append(entropy)

                if acc > data['max_accuracy']:
                    data['max_accuracy'] = acc
                    data['max_accuracy_step'] = i
                    # early stopping (save best model)
                    self.net.save(model_max_output_val=max_val, step=data['max_accuracy_step'])
                    no_change_ctr = 0
                else:
                    no_change_ctr += 1

                self.net.add_accuracy(acc, entropy, i, stage='Validation')
                print ("step {}, max accuracy {}, testing accuracy {}, no change ctr {}, max output val {}".format(i, data['max_accuracy'], acc, no_change_ctr, max_val))

            # UNCOMMENT BEFORE PUSHING TO GITHUB
            # if no_change_ctr == 100:
            #     break

            # perform gradient step
            _, entropy, acc, output_vals, max_value = self.net.train(s_j_batch, a_batch)
            data['training_step'].append(i)
            data['training_accuracy'].append(acc)
            data['training_entropy'].append(entropy)
            if (i % self.eval_freq) == 0:
                print ("\tstep {}, training accuracy {}".format(i, acc))

            self.net.add_accuracy(acc, entropy, i, stage='Training')

            if max_value > max_val:
                max_val = max_value

        self.net.save(model_max_output_val=max_val, relative='/final/')
        pickle.dump(data, open(self.folder + '/data', 'wb'), pickle.HIGHEST_PROTOCOL)
        print ("final max output val {}".format(max_val))

    def save_max_value(self, max_val=-(sys.maxsize)):
        batch = self.replay_memory.size * 10 // 100
        for i in range(100):
            s_j_batch, a_batch, _, _, _ = self.replay_memory.random_batch(batch)
            _, _, output_vals, max_value = self.net.evaluate_batch(s_j_batch, a_batch)
            if i%10 == 0:
                print ("step {}, max output val {}".format(i, max_val))

            if max_value > max_val:
                print ("Max value from {} to {}".format(max_val, max_value))
                max_val = max_value

        print ("max output val {}".format(max_val))
        self.net.save_max_value(max_val)


def classify_demo(args):
    """
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4  --cuda-devices=0 --gpu-fraction=0.4 --optimizer=Adam --lr=0.0001 --decay=0.0 --momentum=0.0 --epsilon=0.001 --train-max-steps=150000 --batch=32 --eval-freq=500 --classify-demo
    """
    #from dqn_net_bn_class import DqnNetClass
    from dqn_net_class import DqnNetClass
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        if args.cuda_devices != '':
            os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    import tensorflow as tf

    if args.path is not None:
        path = args.path
    else:
        path = os.getcwd() + '/'

    if not os.path.exists('results/pretrained_models/dqn'):
        os.makedirs('results/pretrained_models/dqn')

    if args.folder is not None:
        folder = '{}_{}'.format(args.gym_env.replace('-', '_'), args.folder)
    else:
        folder = 'results/pretrained_models/dqn/{}_{}_classifier'.format(args.gym_env.replace('-', '_'), args.optimizer.lower())

    if args.demo_memory_folder is not None:
        demo_memory_folder = args.demo_memory_folder
    else:
        demo_memory_folder = "demo_samples/{}".format(args.gym_env.replace('-', '_'))

    if args.cpu_only:
        device = '/cpu:0'
        gpu_options = None
    else:
        device = '/gpu:'+os.environ["CUDA_VISIBLE_DEVICES"]
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=True
    )


    game_state = GameState(env_id=args.gym_env, display=False, no_op_max=30, human_demo=False, episode_life=True)

    replay_memory = ReplayMemory(
        args.resized_width, args.resized_height,
        np.random.RandomState(),
        max_steps=args.replay_memory,
        phi_length=args.phi_len,
        num_actions=game_state.env.action_space.n,
        wrap_memory=True,
        full_state_size=game_state.clone_full_state().shape[0],
        clip_reward=True)

    DqnNetClass.use_gpu = not args.cpu_only
    net = DqnNetClass(
        args.resized_height, args.resized_width, args.phi_len,
        game_state.env.action_space.n, args.gym_env,
        optimizer=args.optimizer, learning_rate=args.lr,
        epsilon=args.epsilon, decay=args.decay, momentum=args.momentum,
        verbose=args.verbose, path=path, folder=folder, device=device)
    game_state.close()
    del game_state

    sess = tf.Session(config=config, graph=net.graph)
    net.initializer(sess)
    cd = ClassifyDemo(
        net, replay_memory, args.gym_env, args.train_max_steps, args.batch, args.eval_freq,
        demo_memory_folder=demo_memory_folder, folder=folder)
    cd.run()

    sess.close()
