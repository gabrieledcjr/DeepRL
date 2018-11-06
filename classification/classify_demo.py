#!/usr/bin/env python3
import signal
import os
import numpy as np
import sys
import logging

from common.util import load_memory, solve_weight
from common.game_state import GameState

logger = logging.getLogger("classify_demo")

class ClassifyDemo(object):
    use_dropout = False
    def __init__(self, tf, net, name, train_max_steps, batch_size, grad_applier,
        eval_freq=100, demo_memory_folder='', demo_ids=None, folder='', exclude_num_demo_ep=0,
        use_onevsall=False, weighted_cross_entropy=False, device='/cpu:0', clip_norm=None):
        """ Initialize Classifying Human Demo Training """
        self.net = net
        self.name = name
        self.train_max_steps = train_max_steps
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.demo_memory_folder = demo_memory_folder
        self.folder = folder
        self.tf = tf
        self.exclude_num_demo_ep = exclude_num_demo_ep
        self.use_onevsall = use_onevsall
        self.stop_requested = False

        self.demo_memory, actions_ctr, total_rewards = load_memory(
            name=None,
            demo_memory_folder=self.demo_memory_folder,
            demo_ids=demo_ids,
            imgs_normalized=False)

        if weighted_cross_entropy:
            action_freq = [ actions_ctr[a] for a in range(self.demo_memory[0].num_actions) ]
            loss_weight = solve_weight(action_freq)
            logger.debug("loss_weight: {}".format(loss_weight))

            if self.use_onevsall:
                action_freq_onevsall = []
                loss_weight_onevsall = []
                for i in range(self.net._action_size):
                    other_class = sum([action_freq[j] for j in range(self.net._action_size) if i != j])
                    action_freq_onevsall.append([action_freq[i], other_class])
                    loss_weight_onevsall.append(solve_weight(action_freq_onevsall[i]))
                logger.debug("action_freq (one-vs-all): {}".format(action_freq_onevsall))
                logger.debug("loss_weight (one-vs-all): {}".format(loss_weight_onevsall))

            self.net.prepare_loss(class_weights=loss_weight_onevsall if self.use_onevsall else loss_weight)
        else:
            self.net.prepare_loss(class_weights=None)
        self.net.prepare_evaluate()
        self.apply_gradients = self.prepare_compute_gradients(grad_applier, device, clip_norm=clip_norm)

        max_idx, _ = max(total_rewards.items(), key=lambda a: a[1])
        size_max_idx_mem = len(self.demo_memory[max_idx])
        self.test_batch_si = np.zeros(
            (size_max_idx_mem, self.demo_memory[max_idx].height, self.demo_memory[max_idx].width, self.demo_memory[max_idx].phi_length),
            dtype=np.float32)
        if self.use_onevsall:
            self.test_batch_a = np.zeros((self.net.action_size, size_max_idx_mem, 2), dtype=np.float32)
        else:
            self.test_batch_a = np.zeros((size_max_idx_mem, self.net.action_size), dtype=np.float32)

        for i in range(size_max_idx_mem):
            s0, a0, _, _, _, _, _, _ = self.demo_memory[max_idx][i]
            self.test_batch_si[i] = np.copy(s0)
            if self.use_onevsall:
                for n_class in range(self.net._action_size):
                    if a0 == n_class:
                        self.test_batch_a[n_class][i][0] = 1
                    else:
                        self.test_batch_a[n_class][i][1] = 1
            else:
                self.test_batch_a[i][a0] = 1

    def prepare_compute_gradients(self, grad_applier, device, clip_norm=None):
        with self.net.graph.as_default():
            with self.tf.device(device):
                if self.use_onevsall:
                    apply_gradients = []
                    for n_class in range(self.demo_memory[0].num_actions):
                        apply_gradients.append(
                            self._compute_gradients(grad_applier[n_class], self.net.total_loss[n_class], clip_norm))
                else:
                    apply_gradients = self._compute_gradients(grad_applier, self.net.total_loss, clip_norm)

        return apply_gradients

    def _compute_gradients(self, grad_applier, total_loss, clip_norm=None):
        grads_vars = grad_applier.compute_gradients(total_loss)
        grads = []
        params = []
        for p in grads_vars:
            if p[0] == None:
                continue
            grads.append(p[0])
            params.append(p[1])

        if clip_norm is not None:
            grads, _ = self.tf.clip_by_global_norm(grads, clip_norm)

        grads_vars_updates = zip(grads, params)
        return grad_applier.apply_gradients(grads_vars_updates)

    def train(self, sess, summary_op, summary_writer, exclude_bad_state_k=0):
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
        mem_size = len(self.demo_memory) - self.exclude_num_demo_ep
        logger.info("Training with a set of {} demos".format(mem_size))

        for i in range(self.train_max_steps):
            if self.stop_requested:
                break
            batch_si = []
            batch_a = []

            for _ in range(self.batch_size):
                idx = np.random.randint(0, mem_size)
                s, a, _, _ = self.demo_memory[idx].sample2(1, normalize=False, k_bad_states=exclude_bad_state_k)
                batch_si.append(s[0])
                batch_a.append(a[0])

            train_loss, acc, max_value, _ = sess.run(
                [self.net.total_loss, self.net.accuracy, self.net.max_value, self.apply_gradients],
                feed_dict = {
                    self.net.s: batch_si,
                    self.net.a: batch_a} )

            if max_value > self.max_val:
                self.max_val = max_value

            summary = self.tf.Summary()
            summary.value.add(tag='Train_Loss', simple_value=float(train_loss))

            if i % 5000 == 0:
                logger.debug("i={0:} accuracy={1:.4f} loss={2:.4f} max_val={3:}".format(i, acc, train_loss, self.max_val))
                acc = sess.run(
                    self.net.accuracy,
                    feed_dict = {
                        self.net.s: self.test_batch_si,
                        self.net.a: self.test_batch_a} )
                summary.value.add(tag='Accuracy', simple_value=float(acc))

            summary_writer.add_summary(summary, i)
            summary_writer.flush()

    def train_onevsall(self, sess, summary_op, summary_writer, exclude_noop=False, exclude_bad_state_k=0):
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
        self.max_val = [-(sys.maxsize) for _ in range(self.demo_memory[0].num_actions)]
        mem_size = len(self.demo_memory) - self.exclude_num_demo_ep
        logger.info("Training with a set of {} demos".format(mem_size))
        train_class_ctr = [0 for _ in range(self.demo_memory[0].num_actions)]
        for i in range(self.train_max_steps):
            if self.stop_requested:
                break
            batch_si = []
            batch_a = []

            # alternating randomly between classes and reward
            if exclude_noop:
                n_class = np.random.randint(1, self.demo_memory[0].num_actions)
            else:
                n_class = np.random.randint(0, self.demo_memory[0].num_actions)
            train_class_ctr[n_class] += 1

            # train action network branches with logistic regression
            for _ in range(self.batch_size):
                idx = np.random.randint(0, mem_size)
                s, a, _, _ = self.demo_memory[idx].sample2(
                    1, normalize=False,
                    k_bad_states=exclude_bad_state_k,
                    n_class=n_class, onevsall=True)
                batch_si.append(s[0])
                batch_a.append(a[0])

            train_loss, max_value, _ = sess.run(
                [self.net.total_loss[n_class], self.net.max_value[n_class], self.apply_gradients[n_class]],
                feed_dict = {
                    self.net.s: batch_si,
                    self.net.a: batch_a} )

            if max_value > self.max_val[n_class]:
                self.max_val[n_class] = max_value

            summary = self.tf.Summary()
            summary.value.add(tag='Train_Loss/action {}'.format(n_class), simple_value=float(train_loss))

            if i % 5000 == 0:
                logger.debug("i={0:} class={1:} loss={2:.4f} max_val={3:}".format(i, n_class, train_loss, self.max_val))
                logger.debug("branch_ctrs={}".format(train_class_ctr))
                for n in range(self.demo_memory[0].num_actions):
                    acc = sess.run(
                        self.net.accuracy[n],
                        feed_dict = {
                            self.net.s: self.test_batch_si,
                            self.net.a: self.test_batch_a[n]} )
                    summary.value.add(tag='Accuracy/action {}'.format(n), simple_value=float(acc))
                    logger.debug("    class={0:} accuracy={1:.4f}".format(n, acc))

            summary_writer.add_summary(summary, i)
            summary_writer.flush()

        logger.debug("Training stats:")
        for i in range(self.demo_memory[0].num_actions):
            logger.debug("class {} counter={}".format(i, train_class_ctr[i]))

def classify_demo(args):
    '''
    Multi-Class:
    python3 classification/run_experiment.py --gym-env=PongNoFrameskip-v4 --classify-demo --use-mnih-2015 --train-max-steps=150000 --batch_size=32

    MTL One vs All:
    python3 classification/run_experiment.py --gym-env=PongNoFrameskip-v4 --classify-demo --onevsall-mtl --use-mnih-2015 --train-max-steps=150000 --batch_size=32
    '''
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        assert args.cuda_devices != ''
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    import tensorflow as tf

    if args.cpu_only:
        device = "/cpu:0"
        gpu_options = None
    else:
        device = "/gpu:"+os.environ["CUDA_VISIBLE_DEVICES"]
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False)

    if args.demo_memory_folder is not None:
        demo_memory_folder = args.demo_memory_folder
    else:
        demo_memory_folder = 'collected_demo/{}'.format(args.gym_env.replace('-', '_'))

    if args.model_folder is not None:
        model_folder = '{}_{}'.format(args.gym_env.replace('-', '_'), args.model_folder)
    else:
        model_folder = 'results/pretrain_models/{}'.format(args.gym_env.replace('-', '_'))
        end_str = ''
        if args.use_mnih_2015:
            end_str += '_mnih2015'
        if args.onevsall_mtl:
            end_str += '_onevsall_mtl'
        if args.exclude_noop:
            end_str += '_exclude_noop'
        if args.exclude_num_demo_ep > 0:
            end_str += '_exclude{}demoeps'.format(args.exclude_num_demo_ep)
        if args.exclude_k_steps_bad_state > 0:
            end_str += '_exclude{}badstate'.format(args.exclude_k_steps_bad_state)
        if args.l2_beta > 0:
            end_str += '_l2beta{:.0E}'.format(args.l2_beta)
        if args.l1_beta > 0:
            end_str += '_l1beta{:.0E}'.format(args.l1_beta)
        if args.weighted_cross_entropy:
            end_str += '_weighted_loss'
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

    if True:
        from common.util import LogFormatter
        fh = logging.FileHandler('{}/classify.log'.format(model_folder), mode='w')
        fh.setLevel(logging.DEBUG)
        formatter = LogFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logging.getLogger('atari_wrapper').addHandler(fh)
        logging.getLogger('network').addHandler(fh)
        logging.getLogger('deep_rl').addHandler(fh)
        logging.getLogger('replay_memory').addHandler(fh)

    game_state = GameState(env_id=args.gym_env)
    action_size = game_state.env.action_space.n
    game_state.env.close()
    del game_state.env
    del game_state

    if args.onevsall_mtl:
        from network import MTLBinaryClassNetwork
        MTLBinaryClassNetwork.use_mnih_2015 = args.use_mnih_2015
        MTLBinaryClassNetwork.l1_beta = args.l1_beta
        MTLBinaryClassNetwork.l2_beta = args.l2_beta
        MTLBinaryClassNetwork.use_gpu = not args.cpu_only
        network = MTLBinaryClassNetwork(action_size, -1, device)
    else:
        from network import MultiClassNetwork
        MultiClassNetwork.use_mnih_2015 = args.use_mnih_2015
        MultiClassNetwork.l1_beta = args.l1_beta
        MultiClassNetwork.l2_beta = args.l2_beta
        MultiClassNetwork.use_gpu = not args.cpu_only
        network = MultiClassNetwork(action_size, -1, device)

    with tf.device(device):
        if args.onevsall_mtl:
            opt = []
            for n_optimizer in range(action_size):
                opt.append(tf.train.RMSPropOptimizer(
                    learning_rate=args.learn_rate,
                    decay=args.opt_alpha,
                    epsilon=args.opt_epsilon))
        else:
            #opt = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=0.001)
            opt = tf.train.RMSPropOptimizer(
                learning_rate=args.learn_rate,
                decay=args.opt_alpha,
                epsilon=args.opt_epsilon)

    demo_ids = tuple(map(int, args.demo_ids.split(",")))

    classify_demo = ClassifyDemo(
        tf, network, args.gym_env, int(args.train_max_steps),
        args.batch_size, opt, eval_freq=500,
        demo_memory_folder=demo_memory_folder,
        demo_ids=demo_ids,
        folder=model_folder,
        exclude_num_demo_ep=args.exclude_num_demo_ep,
        use_onevsall=args.onevsall_mtl,
        weighted_cross_entropy=args.weighted_cross_entropy,
        device=device, clip_norm=args.grad_norm_clip)

    # prepare session
    sess = tf.Session(config=config, graph=network.graph)

    with network.graph.as_default():
        init = tf.global_variables_initializer()
    sess.run(init)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(model_folder + '/log', sess.graph)

    # init or load checkpoint with saver
    with network.graph.as_default():
        saver = tf.train.Saver()

    def signal_handler(signal, frame):
        nonlocal classify_demo
        logger.info('You pressed Ctrl+C!')
        classify_demo.stop_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    print ('Press Ctrl+C to stop')

    if args.onevsall_mtl:
        classify_demo.train_onevsall(sess, summary_op, summary_writer, exclude_noop=args.exclude_noop, exclude_bad_state_k=args.exclude_k_steps_bad_state)
    else:
        classify_demo.train(sess, summary_op, summary_writer, exclude_bad_state_k=args.exclude_k_steps_bad_state)

    logger.info('Now saving data. Please wait')
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
    logger.info('Data saved!')

    sess.close()
