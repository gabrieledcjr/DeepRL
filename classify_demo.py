# -*- coding: utf-8 -*-
import os
import time
import sqlite3
import numpy as np
import sys

from game_class_network import GameACFFNetwork, GameACLSTMNetwork
from util import get_compressed_images
from data_set import DataSet
from game_state import GameState

try:
    import cPickle as pickle
except ImportError:
    import pickle

class ClassifyDemo(object):
    def __init__(self, tf, net, name, train_max_steps, batch_size, grad_applier,
        eval_freq=100, demo_memory_folder='', folder='', use_lstm=False, device=None):
        """ Initialize Classifying Human Demo Training """
        self.net = net
        self.D = []
        self.name = name
        self.train_max_steps = train_max_steps
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.demo_memory_folder = demo_memory_folder
        self.folder = folder
        self.use_lstm = use_lstm

        self.conn = sqlite3.connect(self.demo_memory_folder + '/demo.db')
        self.db = self.conn.cursor()

        self._load_memory()

        self.net.prepare_loss()
        self.net.prepare_evaluate()
        with tf.device(device):
          var_refs = [v._ref() for v in self.net.get_vars()]
          self.gradients = tf.gradients(
            self.net.total_loss, var_refs,
            gate_gradients=False,
            aggregation_method=None,
            colocate_gradients_with_ops=False)
        grads_vars_updates = zip(self.gradients, self.net.get_vars())
        self.apply_gradients = grad_applier.apply_gradients(grads_vars_updates)

    def _load_memory(self):
        assert os.path.isfile(self.demo_memory_folder + '/demo.db')
        print ("Loading data")
        total_memory = 0
        for demo in self.db.execute("SELECT * FROM demo_samples"):
            print (demo)
            assert demo[2] == self.name
            ep = demo[1]
            total_memory += demo[4]
            folder = self.demo_memory_folder + '/{n:03d}/'.format(n=(ep))
            D = DataSet()
            data = pickle.load(open(folder + self.name + '-dqn.pkl', 'rb'))
            D.width = data['D.width']
            D.height = data['D.height']
            D.max_steps = data['D.max_steps']
            D.phi_length = data['D.phi_length']
            D.num_actions = data['D.num_actions']
            D.actions = data['D.actions']
            D.rewards = data['D.rewards']
            D.terminal = data['D.terminal']
            D.size = data['D.size']
            D.imgs = get_compressed_images(folder + self.name + '-dqn-images.h5' + '.gz')
            self.D.append(D)
        print ("D size: {}".format(len(self.D)))
        print ("Total memory: {}".format(total_memory))
        print ("Data loaded!")

    def run(self, sess, summary_op, summary_writer, accuracy, loss):
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

        D_size = len(self.D)
        for i in range(self.train_max_steps):
            if self.use_lstm:
              start_lstm_state = self.net.lstm_state_out

            idx = np.random.randint(0, D_size)
            batch_si, batch_a, _, _ = self.D[idx].random_batch(self.batch_size)

            if self.use_lstm:
                ls, acc, _ = sess.run(
                    [self.net.total_loss, self.net.accuracy, self.apply_gradients],
                    feed_dict = {
                        self.net.s: batch_si,
                        self.net.a: batch_a,
                        self.net.initial_lstm_state: start_lstm_state,
                        self.net.step_size: [len(batch_a)]} )
            else:
                ls, acc, _ = sess.run(
                    [self.net.total_loss, self.net.accuracy, self.apply_gradients],
                    feed_dict = {
                        self.net.s: batch_si,
                        self.net.a: batch_a} )
            summary_str = sess.run(summary_op, feed_dict={
                loss: ls,
                accuracy: acc,
            })
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

            if i % 100 == 0:
                print ("t={} accuracy={} loss={}".format(i, acc, ls))

def classify_demo(args):
    '''
    python3 run_experiment.py --gym-env=PongDeterministic-v3 --classify-demo --use-lstm --use-mnih-2015
    '''
    if args.use_gpu:
        assert args.cuda_devices != ''
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import tensorflow as tf

    device = "/cpu:0"
    gpu_options = None
    if args.use_gpu:
        device = "/gpu:"+os.environ["CUDA_VISIBLE_DEVICES"]
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)

    if args.demo_memory_folder is not None:
        demo_memory_folder = 'demo_samples/{}'.format(args.demo_memory_folder)
    else:
        demo_memory_folder = 'demo_samples/{}'.format(args.gym_env.replace('-', '_'))

    if args.model_folder is not None:
        model_folder = '{}_{}'.format(args.gym_env.replace('-', '_'), args.model_folder)
    else:
        model_folder = '{}_classifier'.format(args.gym_env.replace('-', '_'))

    end_str = ''
    if args.use_mnih_2015:
        end_str += '_use_mnih'
    if args.use_lstm:
        end_str += '_use_lstm'
    model_folder += end_str

    # assert args.initial_learn_rate > 0
    # initial_learning_rate = args.initial_learn_rate
    # print (colored('Initial Learning Rate={}'.format(initial_learning_rate), 'green'))
    # time.sleep(2)

    game_state = GameState(env_id=args.gym_env)
    action_size = game_state.env.n_actions
    game_state.env.close()
    del game_state.env
    del game_state

    if args.use_lstm:
        GameACLSTMNetwork.use_mnih_2015 = args.use_mnih_2015
        network = GameACLSTMNetwork(action_size, -1, device)
    else:
        GameACFFNetwork.use_mnih_2015 = args.use_mnih_2015
        network = GameACFFNetwork(action_size, -1, device)

    # grad_applier = RMSPropApplier(
    #     learning_rate = initial_learning_rate,
    #     decay = args.rmsp_alpha,
    #     momentum = 0.0,
    #     epsilon = args.rmsp_epsilon,
    #     clip_norm = args.grad_norm_clip,
    #     device = device)
    opt = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=0.001)

    classify_demo = ClassifyDemo(
        tf, network, args.gym_env, int(args.max_time_step),
        args.local_t_max, opt, eval_freq=500,
        demo_memory_folder=demo_memory_folder,
        folder=model_folder, use_lstm=args.use_lstm,
        device=device)

    # prepare session
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
        allow_soft_placement=True)
    sess = tf.Session(config=config)

    init = tf.global_variables_initializer()
    sess.run(init)

    # summary for tensorboard
    accuracy = tf.placeholder(tf.int32)
    loss = tf.placeholder(tf.int32)
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.scalar("loss", loss)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(model_folder + '/log_tb', sess.graph)

    classify_demo.run(sess, summary_op, summary_writer, accuracy, loss)
