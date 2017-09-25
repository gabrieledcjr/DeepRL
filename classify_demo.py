# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import sys

from util import load_memory, solve_weight
from game_state import GameState
from termcolor import colored


class ClassifyDemo(object):
    auto_start = False
    def __init__(self, tf, net, name, train_max_steps, batch_size, grad_applier,
        eval_freq=100, demo_memory_folder='', folder='', use_lstm=False, device=None,
        exclude_num_demo_ep=0, use_onevsall=False):
        """ Initialize Classifying Human Demo Training """
        self.net = net
        self.name = name
        self.train_max_steps = train_max_steps
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.demo_memory_folder = demo_memory_folder
        self.folder = folder
        self.use_lstm = use_lstm
        self.tf = tf
        self.exclude_num_demo_ep = exclude_num_demo_ep
        self.use_onevsall = use_onevsall

        self.D, actions_ctr, _ = load_memory(self.name, self.demo_memory_folder, imgs_normalized=True)

        action_freq = [ actions_ctr[a] for a in range(self.D[0].num_actions) ]
        print ("Action frequency:", action_freq)
        loss_weight = solve_weight(action_freq)
        print ("Class weights:", loss_weight)

        self.net.prepare_loss(class_weights=None)
        self.net.prepare_evaluate()
        self.prepare_compute_gradients(grad_applier)

    def prepare_compute_gradients(self, grad_applier):
        with self.net.graph.as_default():
            if self.use_onevsall:
                self.apply_gradients = []
                for n_class in range(self.D[0].num_actions):
                    grads_vars = grad_applier[n_class].compute_gradients(self.net.total_loss[n_class])
                    grads = []
                    params = []
                    for p in grads_vars:
                        if p[0] == None:
                            continue
                        grads.append(p[0])
                        params.append(p[1])

                    grads_vars_updates = zip(grads, params)
                    self.apply_gradients.append(grad_applier[n_class].apply_gradients(grads_vars_updates))
            else:
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

    def train(self, sess, summary_op, summary_writer):
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
        D_size = len(self.D) - self.exclude_num_demo_ep
        print ("Training with a set of {} demos".format(D_size))

        for i in range(self.train_max_steps):
            batch_si = []
            batch_a = []
            for _ in range(self.batch_size):
                idx = np.random.randint(0, D_size)
                samp_idx = np.random.randint(0, len(self.D[idx]))
                s, ai, _, _ = self.D[idx][samp_idx]
                while self.auto_start:
                    if ai != 1: # Breakout only (FIRE)
                        if ai == 2:
                            ai = 1
                        elif ai == 3:
                            ai = 2
                        break
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

            summary = self.tf.Summary()
            summary.value.add(tag='Accuracy', simple_value=float(acc))
            summary.value.add(tag='Train_Loss', simple_value=float(train_loss))
            summary_writer.add_summary(summary, i)
            summary_writer.flush()

            if i % 500 == 0:
                print ("i={0:} accuracy={1:.4f} loss={2:.4f} max_val={3:}".format(i, acc, train_loss, self.max_val))

    def train_onevsall(self, sess, summary_op, summary_writer, exclude_noop=False):
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
        self.max_val = [-(sys.maxsize) for _ in range(self.D[0].num_actions)]
        D_size = len(self.D) - self.exclude_num_demo_ep
        print ("Training with a set of {} demos".format(D_size))
        train_class_ctr = [0 for _ in range(self.D[0].num_actions)]
        for i in range(self.train_max_steps):
            batch_si = []
            batch_a = []
            # alternating randomly between classes
            if exclude_noop:
                n_class = np.random.randint(1, self.D[0].num_actions)
            else:
                n_class = np.random.randint(0, self.D[0].num_actions)
            train_class_ctr[n_class] += 1
            for _ in range(self.batch_size):
                idx = np.random.randint(0, D_size)
                samp_idx = np.random.randint(0, len(self.D[idx]))
                s, ai, _, _ = self.D[idx][samp_idx]
                while self.auto_start:
                    if ai != 1: # Breakout only (FIRE)
                        if ai == 2:
                            ai = 1
                        elif ai == 3:
                            ai = 2
                        break
                    samp_idx = np.random.randint(0, len(self.D[idx]))
                    s, ai, _, _ = self.D[idx][samp_idx]

                batch_si.append(s)
                a = np.zeros([2])
                if ai == n_class:
                    a[0] = 1
                else:
                    a[1] = 1
                batch_a.append(a)
            # idx = np.random.randint(0, D_size)
            # batch_si, batch_a, _, _ = self.D[idx].random_batch(self.batch_size, normalize=True)

            train_loss, acc, max_value, _ = sess.run(
                [self.net.total_loss[n_class], self.net.accuracy[n_class], self.net.max_value[n_class], self.apply_gradients[n_class]],
                feed_dict = {
                    self.net.s: batch_si,
                    self.net.a: batch_a} )

            if max_value > self.max_val[n_class]:
                self.max_val[n_class] = max_value

            summary = self.tf.Summary()
            summary.value.add(tag='Accuracy/action {}'.format(n_class), simple_value=float(acc))
            summary.value.add(tag='Train_Loss/action {}'.format(n_class), simple_value=float(train_loss))
            summary_writer.add_summary(summary, i)
            summary_writer.flush()

            if i % 500 == 0:
                print ("i={0:} class={1:} accuracy={2:.4f} loss={3:.4f} max_val={4:}".format(i, n_class, acc, train_loss, self.max_val))

        print ("Training stats:")
        for i in range(self.D[0].num_actions):
            print("Class {} counter={}".format(i, train_class_ctr[i]))

def classify_demo(args):
    '''
    Multi-Class:
    python3 run_experiment.py --gym-env=PongDeterministic-v3 --classify-demo --use-mnih-2015 --max-time-step=150000 --local-t-max=32

    MTL One vs All:
    python3 run_experiment.py --gym-env=PongDeterministic-v3 --classify-demo --onevsall-mtl --use-mnih-2015 --max-time-step=150000 --local-t-max=32
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
        demo_memory_folder = args.demo_memory_folder
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
        if args.auto_start:
            end_str += '_autostart'
        if args.onevsall_mtl:
            end_str += '_onevsall_mtl'
        if args.exclude_noop:
            end_str += '_exclude_noop'
        if args.exclude_num_demo_ep > 0:
            end_str += '_exclude{}demoeps'.format(args.exclude_num_demo_ep)
        model_folder += end_str

    if args.append_experiment_num is not None:
        model_folder += '_' + args.append_experiment_num

    if not os.path.exists(model_folder + '/transfer_model'):
        os.makedirs(model_folder + '/transfer_model')
        os.makedirs(model_folder + '/transfer_model/all')
        os.makedirs(model_folder + '/transfer_model/nofc2')
        os.makedirs(model_folder + '/transfer_model/nofc1')
        if args.use_mnih_2015:
            os.makedirs(model_folder + '/transfer_model/noconv3')
        os.makedirs(model_folder + '/transfer_model/noconv2')

    # assert args.initial_learn_rate > 0
    # initial_learning_rate = args.initial_learn_rate
    # print (colored('Initial Learning Rate={}'.format(initial_learning_rate), 'green'))
    # time.sleep(2)

    game_state = GameState(env_id=args.gym_env)
    action_size = game_state.env.n_actions
    if args.auto_start:
        action_size -= 1
    game_state.env.close()
    del game_state.env
    del game_state

    if args.use_lstm:
        print ("Can't use lstm")

    if args.onevsall_mtl:
        from game_class_network import MTLBinaryClassNetwork
        MTLBinaryClassNetwork.use_mnih_2015 = args.use_mnih_2015
        MTLBinaryClassNetwork.l1_beta = args.l1_beta
        MTLBinaryClassNetwork.l2_beta = args.l2_beta
        network = MTLBinaryClassNetwork(action_size, -1, device)
    else:
        from game_class_network import MultiClassNetwork
        MultiClassNetwork.use_mnih_2015 = args.use_mnih_2015
        MultiClassNetwork.l1_beta = args.l1_beta
        MultiClassNetwork.l2_beta = args.l2_beta
        network = MultiClassNetwork(action_size, -1, device)

    # grad_applier = RMSPropApplier(
    #     learning_rate = initial_learning_rate,
    #     decay = args.rmsp_alpha,
    #     momentum = 0.0,
    #     epsilon = args.rmsp_epsilon,
    #     clip_norm = args.grad_norm_clip,
    #     device = device)
    with tf.device(device):
        if args.onevsall_mtl:
            opt = []
            for n_optimizer in range(action_size):
                opt.append(tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=0.001))
        else:
            opt = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=0.001)

    ClassifyDemo.auto_start = args.auto_start
    classify_demo = ClassifyDemo(
        tf, network, args.gym_env, int(args.max_time_step),
        args.local_t_max, opt, eval_freq=500,
        demo_memory_folder=demo_memory_folder,
        folder=model_folder, use_lstm=args.use_lstm,
        device=device, exclude_num_demo_ep=args.exclude_num_demo_ep,
        use_onevsall=args.onevsall_mtl)

    # prepare session
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
        allow_soft_placement=True)
    sess = tf.Session(config=config, graph=network.graph)

    with network.graph.as_default():
        init = tf.global_variables_initializer()
    sess.run(init)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(model_folder + '/log_tb', sess.graph)

    # init or load checkpoint with saver
    with network.graph.as_default():
        saver = tf.train.Saver()

    if args.onevsall_mtl:
        classify_demo.train_onevsall(sess, summary_op, summary_writer, exclude_noop=args.exclude_noop)
    else:
        classify_demo.train(sess, summary_op, summary_writer)

    print(colored('Now saving data. Please wait', 'yellow'))
    saver.save(sess, model_folder + '/' + '{}_checkpoint'.format(args.gym_env.replace('-', '_')))

    with network.graph.as_default():
        transfer_params = tf.get_collection("transfer_params")
        transfer_saver = tf.train.Saver(transfer_params)
    transfer_saver.save(sess, model_folder + '/transfer_model/all/' + '{}_transfer_params'.format(args.gym_env.replace('-', '_')))

    # Remove fc2 weights
    for param in transfer_params[:]:
        if param.op.name == "net_-1/fc2_weights" or param.op.name == "net_-1/fc2_biases":
            transfer_params.remove(param)

    with network.graph.as_default():
        transfer_saver = tf.train.Saver(transfer_params)
    transfer_saver.save(sess, model_folder + '/transfer_model/nofc2/' + '{}_transfer_params'.format(args.gym_env.replace('-', '_')))

    # Remove fc1 weights
    for param in transfer_params[:]:
        if param.op.name == "net_-1/fc1_weights" or param.op.name == "net_-1/fc1_biases":
            transfer_params.remove(param)

    with network.graph.as_default():
        transfer_saver = tf.train.Saver(transfer_params)
    transfer_saver.save(sess, model_folder + '/transfer_model/nofc1/' + '{}_transfer_params'.format(args.gym_env.replace('-', '_')))

    # Remove conv3 weights
    if args.use_mnih_2015:
        for param in transfer_params[:]:
            if param.op.name == "net_-1/conv3_weights" or param.op.name == "net_-1/conv3_biases":
                transfer_params.remove(param)

        with network.graph.as_default():
            transfer_saver = tf.train.Saver(transfer_params)
        transfer_saver.save(sess, model_folder + '/transfer_model/noconv3/' + '{}_transfer_params'.format(args.gym_env.replace('-', '_')))

    # Remove conv2 weights
    for param in transfer_params[:]:
        if param.op.name == "net_-1/conv2_weights" or param.op.name == "net_-1/conv2_biases":
            transfer_params.remove(param)

    with network.graph.as_default():
        transfer_saver = tf.train.Saver(transfer_params)
    transfer_saver.save(sess, model_folder + '/transfer_model/noconv2/' + '{}_transfer_params'.format(args.gym_env.replace('-', '_')))

    with open(model_folder + '/transfer_model/max_output_value', 'w') as f_max_value:
        f_max_value.write(str(classify_demo.max_val))
    print (colored('Data saved!', 'green'))
