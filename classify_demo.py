# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import sys

from game_class_network import GameACFFNetwork, GameACLSTMNetwork
from util import load_memory
from game_state import GameState
from termcolor import colored


class ClassifyDemo(object):
    def __init__(self, tf, net, name, train_max_steps, batch_size, grad_applier,
        eval_freq=100, demo_memory_folder='', folder='', use_lstm=False, device=None):
        """ Initialize Classifying Human Demo Training """
        self.net = net
        self.name = name
        self.train_max_steps = train_max_steps
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.demo_memory_folder = demo_memory_folder
        self.folder = folder
        self.use_lstm = use_lstm

        self.D, actions_ctr = load_memory(self.name, self.demo_memory_folder, imgs_normalized=True)

        self.net.prepare_loss()
        self.net.prepare_evaluate()
        grads_vars = grad_applier.compute_gradients(self.net.total_loss)
        grads = []
        params = []
        for p in grads_vars:
            if p[0] == None:
                continue
            grads.append(p[0])
            params.append(p[1])

        #grads = tf.clip_by_global_norm(grads, 1)[0]
        grads_vars_updates = zip(grads, params)
        self.apply_gradients = grad_applier.apply_gradients(grads_vars_updates)
        #self.apply_gradients = grad_applier.minimize(self.net.total_loss)

    def run_ff(self, sess, summary_op, summary_writer, accuracy, loss):
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
        self.max_val = -(sys.maxsize)
        D_size = len(self.D)

        for i in range(self.train_max_steps):
            batch_si = []
            batch_a = []
            for _ in range(self.batch_size):
                idx = np.random.randint(0, D_size)
                samp_idx = np.random.randint(0, len(self.D[idx]))
                s, ai, _, _ = self.D[idx][samp_idx]
                batch_si.append(s)
                a = np.zeros([self.net._action_size])
                a[ai] = 1
                batch_a.append(a)
            # idx = np.random.randint(0, D_size)
            # batch_si, batch_a, _, _ = self.D[idx].random_batch(self.batch_size, normalize=True)
            train_loss, acc, max_value, _ = sess.run(
                [self.net.total_loss, self.net.accuracy, self.net.max_value, self.apply_gradients],
                feed_dict = {
                    self.net.s: batch_si,
                    self.net.a: batch_a} )

            if max_value > self.max_val:
                self.max_val = max_value

            summary_str = sess.run(summary_op, feed_dict={
                loss: train_loss,
                accuracy: acc,
            })
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

            if i % 500 == 0:
                print ("i={} accuracy={} loss={} max_val={}".format(i, acc, train_loss, self.max_val))

    def run_lstm(self, sess, summary_op, summary_writer, accuracy, loss):
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
        max_val = -(sys.maxsize)

        D_size = len(self.D)
        terminal = False
        get_index = True
        epoch = 0
        for i in range(self.train_max_steps):
            if get_index:
                idx = np.random.randint(0, D_size)
                get_index = False
                last_index = 0
                training_loss = 0.
                steps = 0
                accuracy_accum = 0.
                start_lstm_state = self.net.lstm_state_out

            batch_si = []
            batch_a = []
            count_batch = 0
            while True:
                s, ai, _, terminal = self.D[idx][last_index]
                batch_si.append(s * (1.0/255.0))
                a = np.zeros([self.net._action_size])
                a[ai] = 1
                batch_a.append(a)
                last_index += 1
                count_batch += 1

                if count_batch == self.batch_size: break
                if len(self.D[idx]) == last_index:
                    self.net.reset_state()
                    get_index = True
                    break

            ls, acc, start_lstm_state, _ = sess.run(
                [self.net.total_loss, self.net.accuracy, self.net.lstm_state, self.apply_gradients],
                feed_dict = {
                    self.net.s: batch_si,
                    self.net.a: batch_a,
                    self.net.initial_lstm_state: start_lstm_state,
                    self.net.step_size: [len(batch_a)]} )

            training_loss += ls
            accuracy_accum += acc
            steps += 1
            if get_index:
                accuracy_accum /= steps
                training_loss /= steps
                summary_str = sess.run(summary_op, feed_dict={
                    loss: training_loss,
                    accuracy: accuracy_accum,
                })
                summary_writer.add_summary(summary_str, epoch)
                summary_writer.flush()

                print ("epoch={} accuracy={} loss={}".format(epoch, accuracy_accum, training_loss))
                epoch += 1

def classify_demo(args):
    '''
    python3 run_experiment.py --gym-env=PongDeterministic-v3 --classify-demo --use-mnih-2015 --max-time-step=150000 --local-t-max=32
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

    if not os.path.exists(model_folder + '/transfer_model'):
        os.makedirs(model_folder + '/transfer_model')
        os.makedirs(model_folder + '/transfer_model/all')
        os.makedirs(model_folder + '/transfer_model/nofc2')

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
    with tf.device(device):
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
    accuracy = tf.placeholder(tf.float32)
    loss = tf.placeholder(tf.float32)
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.scalar("loss", loss)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(model_folder + '/log_tb', sess.graph)

    # init or load checkpoint with saver
    saver = tf.train.Saver()

    if args.use_lstm:
        # FIXME: Not working correctly
        classify_demo.run_lstm(sess, summary_op, summary_writer, accuracy, loss)
    else:
        classify_demo.run_ff(sess, summary_op, summary_writer, accuracy, loss)

    print(colored('Now saving data. Please wait', 'yellow'))
    saver.save(sess, model_folder + '/' + '{}_checkpoint'.format(args.gym_env.replace('-', '_')))

    transfer_params = tf.get_collection("transfer_params")
    transfer_saver = tf.train.Saver(transfer_params)
    transfer_saver.save(sess, model_folder + '/transfer_model/all/' + '{}_transfer_params'.format(args.gym_env.replace('-', '_')))

    # Remove fc2 weights
    for param in transfer_params[:]:
        if param.op.name == "net_-1/fc2_weights" or param.op.name == "net_-1/fc2_biases":
            transfer_params.remove(param)

    transfer_saver_nofc2 = tf.train.Saver(transfer_params)
    transfer_saver_nofc2.save(sess, model_folder + '/transfer_model/nofc2/' + '{}_transfer_params'.format(args.gym_env.replace('-', '_')))

    with open(model_folder + '/transfer_model/max_output_value', 'w') as f_max_value:
        f_max_value.write(str(classify_demo.max_val))
    print (colored('Data saved!', 'green'))
