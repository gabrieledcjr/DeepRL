#!/usr/bin/env python3
"""Supervised learning as pre-training method.

This module uses supervised learning as a pre-training method to learn features
using data from human demonstrations. There are two training options:
1) regular multi-class classification, and 2) one class vs. all class using
multi-task learning technique of having multiple output layers for each class.

Usage:
    Multi-class classification
        $ python3 pretrain/run_experiment.py
            --gym-env=PongNoFrameskip-v4
            --classify-demo --use-mnih-2015
            --train-max-steps=150000 --batch_size=32

    MTL one class vs. all classes
        $ python3 pretrain/run_experiment.py
            --gym-env=PongNoFrameskip-v4
            --classify-demo --onevsall-mtl --use-mnih-2015
            --train-max-steps=150000 --batch_size=32

"""
import cv2
import logging
import numpy as np
import os
import pathlib
import signal
import sys

from common.game_state import GameState
from common.game_state import get_wrapper_by_name
from common.replay_memory import ReplayMemoryReturns
from common.util import compute_proportions
from common.util import load_memory
from common.util import LogFormatter
from common.util import percent_decrease
from common.util import solve_weight
from network import MTLBinaryClassNetwork
from network import MultiClassNetwork
from termcolor import colored

logger = logging.getLogger("classify_demo")


class ClassifyDemo(object):
    """Use Supervised learning for learning features."""

    def __init__(self, tf, net, name, train_max_steps, batch_size,
                 grad_applier, eval_freq=5000, demo_memory_folder=None,
                 demo_ids=None, folder='', exclude_num_demo_ep=0,
                 use_onevsall=False, device='/cpu:0', clip_norm=None,
                 game_state=None, use_batch_proportion=False):
        """Initialize ClassifyDemo class."""
        assert demo_ids is not None
        assert game_state is not None

        self.net = net
        self.name = name
        self.train_max_steps = train_max_steps
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.folder = folder
        self.tf = tf
        self.exclude_num_demo_ep = exclude_num_demo_ep
        self.use_onevsall = use_onevsall
        self.stop_requested = False
        self.game_state = game_state
        self.best_model_reward = -(sys.maxsize)
        self.use_batch_proportion = use_batch_proportion

        logger.info("train_max_steps: {}".format(self.train_max_steps))
        logger.info("batch_size: {}".format(self.batch_size))
        logger.info("eval_freq: {}".format(self.eval_freq))
        logger.info("use_onevsall: {}".format(self.use_onevsall))
        logger.info("use_batch_proportion: {}".format(
            self.use_batch_proportion))

        self.demo_memory, actions_ctr, total_rewards, total_steps = \
            load_memory(name=None, demo_memory_folder=demo_memory_folder,
                        demo_ids=demo_ids, imgs_normalized=False)

        action_freq = [actions_ctr[a] for a in range(self.net.action_size)]

        if self.use_batch_proportion:
            self.batch_proportion = compute_proportions(
                self.batch_size, action_freq)
            logger.info("batch_proportion: {}".format(self.batch_proportion))

        self.net.prepare_loss(sl_loss_weight=1.0, critic_weight=0.01)

        self.net.prepare_evaluate()
        self.apply_gradients = self.prepare_compute_gradients(
            grad_applier, device, clip_norm=clip_norm)

        max_idx, _ = max(total_rewards.items(), key=lambda a: a[1])
        size_max_idx_mem = len(self.demo_memory[max_idx])
        self.test_batch_si = np.zeros(
            (size_max_idx_mem, self.net.in_shape[0], self.net.in_shape[1],
             self.demo_memory[max_idx].phi_length), dtype=np.float32)

        if self.use_onevsall:
            self.test_batch_a = np.zeros(
                (self.net.action_size, size_max_idx_mem, 2), dtype=np.float32)
        else:
            self.test_batch_a = np.zeros(
                (size_max_idx_mem, self.net.action_size), dtype=np.float32)

        for i in range(size_max_idx_mem):
            s0, a0, _, _, _, _, _, _ = self.demo_memory[max_idx][i]
            self.test_batch_si[i] = cv2.resize(s0, self.net.in_shape[:-1],
                                               interpolation=cv2.INTER_AREA)
            if self.use_onevsall:
                for n_class in range(self.net._action_size):
                    if a0 == n_class:
                        self.test_batch_a[n_class][i][0] = 1
                    else:
                        self.test_batch_a[n_class][i][1] = 1
            else:
                self.test_batch_a[i][a0] = 1

        self.combined_memory = ReplayMemoryReturns(
            height=self.net.in_shape[0],
            width=self.net.in_shape[1],
            max_steps=total_steps,
            phi_length=self.demo_memory[max_idx].phi_length,
            num_actions=self.demo_memory[max_idx].num_actions,
            wrap_memory=False,
            full_state_size=self.demo_memory[max_idx].full_state_size)

        returns = np.array([], dtype=np.float32)
        for idx in list(self.demo_memory.keys()):
            demo = self.demo_memory[idx]
            exp_ret = ReplayMemoryReturns.compute_returns(
                demo.rewards, demo.terminal, 0.99, clip=False)
            returns = np.append(returns, exp_ret)

            for i in range(demo.max_steps):
                resize_img = cv2.resize(demo.imgs[i], self.net.in_shape[:-1],
                                        interpolation=cv2.INTER_AREA)
                self.combined_memory.add(resize_img, demo.actions[i],
                                         demo.rewards[i], demo.terminal[i],
                                         demo.lives[i], demo.full_state[i])

            self.combined_memory.returns = returns
            demo.close()
            del demo

    def prepare_compute_gradients(self, grad_applier, device, clip_norm=None):
        """Return operation for gradient application.

        Keyword arguments:
        grad_applier -- optimizer for applying gradients
        device -- cpu or gpu
        clip_norm -- value for clip_by_global_norm (default None)
        """
        with self.net.graph.as_default():
            with self.tf.device(device):
                if self.use_onevsall:
                    apply_gradients = []
                    for n_class in range(self.net.action_size):
                        apply_gradients.append(self.__compute_gradients(
                            grad_applier[n_class],
                            self.net.total_loss[n_class], clip_norm))
                else:
                    apply_gradients = self.__compute_gradients(
                        grad_applier, self.net.total_loss, clip_norm)

        return apply_gradients

    def __compute_gradients(self, grad_applier, total_loss, clip_norm=None):
        """Apply gradient clipping and return op for gradient application."""
        grads_vars = grad_applier.compute_gradients(total_loss)
        grads = []
        params = []
        for p in grads_vars:
            if p[0] is None:
                continue
            grads.append(p[0])
            params.append(p[1])

        if clip_norm is not None:
            grads, _ = self.tf.clip_by_global_norm(grads, clip_norm)

        grads_vars_updates = zip(grads, params)
        return grad_applier.apply_gradients(grads_vars_updates)

    def save_best_model(self, test_reward, best_saver, sess):
        """Save best network model's parameters and reward to file.

        Keyword arguments:
        test_reward -- testing total average reward
        best_saver -- tf saver object
        sess -- tf session
        """
        self.best_model_reward = test_reward
        best_model_reward_file = self.folder / 'model_best/best_model_reward'
        with best_model_reward_file.open('w') as f:
            f.write(str(self.best_model_reward))

        best_file = self.folder / 'model_best'
        best_file /= '{}_checkpoint'.format(self.name.replace('-', '_'))
        best_saver.save(sess, str(best_file))

    def choose_action_with_high_confidence(self, pi_values, exclude_noop=True):
        """Save best network model's parameters and reward to file.

        Keyword arguments:
        test_reward -- testing total average reward
        best_saver -- tf saver object
        sess -- tf session
        """
        max_confidence_action = np.argmax(pi_values[1 if exclude_noop else 0:])
        confidence = pi_values[max_confidence_action]
        return (max_confidence_action+(1 if exclude_noop else 0)), confidence

    def test_game(self, sess):
        """Evaluate game with current network model.

        Keyword argument:
        sess -- tf session
        """
        self.game_state.reset(hard_reset=True)

        max_steps = 25000
        total_reward = 0
        total_steps = 0
        episode_reward = 0
        episode_steps = 0
        n_episodes = 0
        while max_steps > 0:
            model_pi = self.net.run_policy(sess, self.game_state.s_t)
            action, confidence = self.choose_action_with_high_confidence(
                model_pi, exclude_noop=False)

            # take action
            self.game_state.step(action)
            terminal = self.game_state.terminal
            episode_reward += self.game_state.reward
            episode_steps += 1
            max_steps -= 1

            # s_t = s_t1
            self.game_state.update()

            if terminal:
                was_real_done = get_wrapper_by_name(
                    self.game_state.env, 'EpisodicLifeEnv').was_real_done

                if was_real_done:
                    n_episodes += 1
                    score_str = colored("score={}".format(
                        episode_reward), "magenta")
                    steps_str = colored("steps={}".format(
                        episode_steps), "blue")
                    log_data = (n_episodes, score_str, steps_str, total_steps)
                    # logger.debug("test: trial={} {} {} total_steps={}"
                    #              .format(*log_data))
                    total_reward += episode_reward
                    total_steps += episode_steps
                    episode_reward = 0
                    episode_steps = 0

                self.game_state.reset(hard_reset=False)

        if n_episodes == 0:
            total_reward = episode_reward
            total_steps = episode_steps
        else:
            total_reward = total_reward / n_episodes
            total_steps = total_steps // n_episodes

        log_data = (total_reward, total_steps, n_episodes)
        logger.info("test: final score={} final steps={} # trials={}"
                    .format(*log_data))
        return log_data

    def train(self, sess, summary_op, summary_writer, best_saver=None):
        """Train classification with human demonstration.

        This method either does classification training after autoencoder
        training or jointly trains autoencoder and classification using
        Supervised Autoencoder (SAE).

        Keyword arguments:
        sess -- tf session
        summary_op -- tf summary operation
        summary_writer -- tf summary writer
        best_saver -- tf saver for best model
        """
        self.max_val = -(sys.maxsize)
        prev_loss = 0

        for i in range(self.train_max_steps + 1):
            if self.stop_requested:
                break

            if self.use_batch_proportion:
                batch_si, batch_a, _, _, batch_returns = \
                    self.combined_memory.sample_proportional(
                        self.batch_size, self.batch_proportion)
            else:
                batch_si, batch_a, _, _, batch_returns = \
                    self.combined_memory.sample2(self.batch_size)

            feed_dict = {self.net.s: batch_si, self.net.a: batch_a}

            if self.net.use_sil:
                feed_dict[self.net.returns] = batch_returns

            train_loss, acc, max_value, _ = sess.run([
                self.net.total_loss,
                self.net.accuracy,
                self.net.max_value,
                self.apply_gradients],
                feed_dict=feed_dict)

            if max_value > self.max_val:
                self.max_val = max_value

            percent_change = percent_decrease(train_loss, prev_loss)
            if i % self.eval_freq == 0 or percent_change > 2.5:
                prev_loss = train_loss

                test_acc = sess.run(
                    self.net.accuracy,
                    feed_dict={
                        self.net.s: self.test_batch_si,
                        self.net.a: self.test_batch_a})

                summary = self.tf.Summary()
                summary.value.add(tag='Train_Loss',
                                  simple_value=float(train_loss))
                summary.value.add(tag='Accuracy', simple_value=float(acc))

                logger.debug("i={0:} train_acc={1:.4f} test_acc={2:.4f}"
                             " loss={3:.4f} max_val={4:}".format(
                              i, acc, test_acc, train_loss, self.max_val))

                if False:
                    total_reward, total_steps, n_episodes = \
                        self.test_game(sess)
                    summary.value.add(tag='Reward', simple_value=total_reward)
                    summary.value.add(tag='Steps', simple_value=total_steps)
                    summary.value.add(tag='Episodes', simple_value=n_episodes)

                    if total_reward >= self.best_model_reward:
                        self.save_best_model(total_reward, best_saver, sess)

                summary_writer.add_summary(summary, i)
                summary_writer.flush()

    def train_onevsall(self, sess, summary_op, summary_writer,
                       exclude_noop=False, best_saver=None):
        """Train classification with human demonstration (one vs. all).

        This method either does classification training using one class versus
        all classes.

        Keyword arguments:
        sess -- tf session
        summary_op -- tf summary operation
        summary_writer -- tf summary writer
        best_saver -- tf saver for best model
        """
        assert not self.net.use_sil

        self.max_val = [-(sys.maxsize) for _ in range(self.net.action_size)]
        train_class_ctr = [0 for _ in range(self.net.action_size)]
        for i in range(self.train_max_steps + 1):
            if self.stop_requested:
                break

            # alternating randomly between classes and reward
            if exclude_noop:
                n_class = np.random.randint(1, self.net.action_size)
            else:
                n_class = np.random.randint(0, self.net.action_size)
            train_class_ctr[n_class] += 1

            # train action network branches with logistic regression
            batch_si, batch_a, _, _ = self.combined_memory.sample2(
                self.batch_size, normalize=False, n_class=n_class,
                onevsall=True)

            train_loss, max_value, _ = sess.run(
                [self.net.total_loss[n_class], self.net.max_value[n_class],
                 self.apply_gradients[n_class]],
                feed_dict={self.net.s: batch_si, self.net.a: batch_a})

            if max_value > self.max_val[n_class]:
                self.max_val[n_class] = max_value

            summary = self.tf.Summary()
            summary.value.add(tag='Train_Loss/action {}'.format(n_class),
                              simple_value=float(train_loss))

            if i % self.eval_freq == 0:
                logger.debug("i={0:} class={1:} loss={2:.4f} max_val={3:}"
                             .format(i, n_class, train_loss, self.max_val))
                logger.debug("branch_ctrs={}".format(train_class_ctr))

                for n in range(self.net.action_size):
                    acc = sess.run(
                        self.net.accuracy[n],
                        feed_dict={
                            self.net.s: self.test_batch_si,
                            self.net.a: self.test_batch_a[n]})
                    summary.value.add(tag='Accuracy/action {}'.format(n),
                                      simple_value=float(acc))
                    logger.debug("    class={0:} accuracy={1:.4f}"
                                 .format(n, acc))

            summary_writer.add_summary(summary, i)
            summary_writer.flush()

        logger.debug("Training stats:")
        for i in range(self.net.action_size):
            logger.debug("class {} counter={}".format(i, train_class_ctr[i]))


def classify_demo(args):
    """Use supervised learning to learn features from human demo."""
    GYM_ENV_NAME = args.gym_env.replace('-', '_')

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
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=args.gpu_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False)

    if args.demo_memory_folder is not None:
        demo_memory_folder = args.demo_memory_folder
    else:
        demo_memory_folder = 'collected_demo/{}'.format(GYM_ENV_NAME)

    demo_memory_folder = pathlib.Path(demo_memory_folder)

    if args.model_folder is not None:
        model_folder = '{}_{}'.format(GYM_ENV_NAME, args.model_folder)
    else:
        model_folder = 'results/pretrain_models/{}'.format(GYM_ENV_NAME)
        end_str = ''
        if args.use_mnih_2015:
            end_str += '_mnih2015'
        if args.padding == 'SAME':
            end_str += '_same'
        if args.optimizer == 'adam':
            end_str += '_adam'
        if args.onevsall_mtl:
            end_str += '_onevsall_mtl'
        if args.exclude_noop:
            end_str += '_exclude_noop'
        if args.exclude_num_demo_ep > 0:
            end_str += '_exclude{}demoeps'.format(args.exclude_num_demo_ep)
        if args.l2_beta > 0:
            end_str += '_l2beta{:.0E}'.format(args.l2_beta)
        if args.l1_beta > 0:
            end_str += '_l1beta{:.0E}'.format(args.l1_beta)
        if args.grad_norm_clip is not None:
            end_str += '_clipnorm{:.0E}'.format(args.grad_norm_clip)
        if args.use_batch_proportion:
            end_str += '_batchprop'
        if args.use_sil:
            end_str += '_sil'
        model_folder += end_str

    if args.append_experiment_num is not None:
        model_folder += '_' + args.append_experiment_num

    model_folder = pathlib.Path(model_folder)

    if not (model_folder / 'transfer_model').exists():
        os.makedirs(str(model_folder / 'transfer_model'))
        os.makedirs(str(model_folder / 'transfer_model/all'))
        os.makedirs(str(model_folder / 'transfer_model/nofc2'))
        os.makedirs(str(model_folder / 'transfer_model/nofc1'))
        if args.use_mnih_2015:
            os.makedirs(str(model_folder / 'transfer_model/noconv3'))
        os.makedirs(str(model_folder / 'transfer_model/noconv2'))
        os.makedirs(str(model_folder / 'model_best'))

    fh = logging.FileHandler(str(model_folder / 'classify.log'), mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = LogFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logging.getLogger('atari_wrapper').addHandler(fh)
    logging.getLogger('network').addHandler(fh)
    logging.getLogger('deep_rl').addHandler(fh)
    logging.getLogger('replay_memory').addHandler(fh)

    game_state = GameState(env_id=args.gym_env)
    action_size = game_state.env.action_space.n

    input_shape = (args.input_shape, args.input_shape, 4)
    if args.onevsall_mtl:
        MTLBinaryClassNetwork.use_mnih_2015 = args.use_mnih_2015
        MTLBinaryClassNetwork.l1_beta = args.l1_beta
        MTLBinaryClassNetwork.l2_beta = args.l2_beta
        MTLBinaryClassNetwork.use_gpu = not args.cpu_only
        network = MTLBinaryClassNetwork(action_size, -1, device)
    else:
        MultiClassNetwork.use_mnih_2015 = args.use_mnih_2015
        MultiClassNetwork.l1_beta = args.l1_beta
        MultiClassNetwork.l2_beta = args.l2_beta
        MultiClassNetwork.use_gpu = not args.cpu_only
        network = MultiClassNetwork(
            action_size, -1, device, padding=args.padding,
            in_shape=input_shape)

    logger.info("optimizer: {}".format(
        'RMSPropOptimizer' if args.optimizer == 'rms' else 'AdamOptimizer'))
    logger.info("\tlearning_rate: {}".format(args.learn_rate))
    logger.info("\tepsilon: {}".format(args.opt_epsilon))
    if args.optimizer == 'rms':
        logger.info("\tdecay: {}".format(args.opt_alpha))
    else:  # Adam
        # Tensorflow defaults
        beta1 = 0.9
        beta2 = 0.999

    with tf.device(device):
        if args.onevsall_mtl:
            opt = []
            for n_optimizer in range(action_size):
                if args.optimizer == 'rms':
                    opt.append(tf.train.RMSPropOptimizer(
                        learning_rate=args.learn_rate,
                        decay=args.opt_alpha,
                        epsilon=args.opt_epsilon))
                else:
                    opt.append(tf.train.AdamOptimizer(
                        learning_rate=args.learn_rate,
                        beta1=beta1, beta2=beta2,
                        epsilon=args.opt_epsilon))

        else:
            if args.optimizer == 'rms':
                opt = tf.train.RMSPropOptimizer(
                    learning_rate=args.learn_rate,
                    decay=args.opt_alpha,
                    epsilon=args.opt_epsilon)

            else:  # Adam
                opt = tf.train.AdamOptimizer(
                    learning_rate=args.learn_rate,
                    beta1=beta1, beta2=beta2,
                    epsilon=args.opt_epsilon)

    classify_demo = ClassifyDemo(
        tf, network, args.gym_env, int(args.train_max_steps),
        args.batch_size, opt, eval_freq=args.eval_freq,
        demo_memory_folder=demo_memory_folder,
        demo_ids=args.demo_ids,
        folder=model_folder,
        exclude_num_demo_ep=args.exclude_num_demo_ep,
        use_onevsall=args.onevsall_mtl,
        device=device, clip_norm=args.grad_norm_clip,
        game_state=game_state,
        use_batch_proportion=args.use_batch_proportion)

    # prepare session
    sess = tf.Session(config=config, graph=network.graph)

    with network.graph.as_default():
        init = tf.global_variables_initializer()
    sess.run(init)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(str(model_folder / 'log'),
                                           sess.graph)

    # init or load checkpoint with saver
    with network.graph.as_default():
        saver = tf.train.Saver()
        best_saver = tf.train.Saver(max_to_keep=1)

    def signal_handler(signal, frame):
        nonlocal classify_demo
        logger.info('You pressed Ctrl+C!')
        classify_demo.stop_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C to stop')

    if args.onevsall_mtl:
        classify_demo.train_onevsall(sess, summary_op, summary_writer,
                                     exclude_noop=args.exclude_noop,
                                     best_saver=best_saver)
    else:
        classify_demo.train(sess, summary_op, summary_writer,
                            best_saver=best_saver)

    logger.info('Now saving data. Please wait')
    saver.save(sess, str(model_folder / '{}_checkpoint'.format(GYM_ENV_NAME)))

    with network.graph.as_default():
        transfer_params = tf.get_collection("transfer_params")
        transfer_saver = tf.train.Saver(transfer_params)
    transfer_saver.save(
        sess, str(model_folder / 'transfer_model/all'
                  / '{}_transfer_params'.format(GYM_ENV_NAME)))

    # Remove fc2/fc3 weights
    for param in transfer_params[:]:
        name = param.op.name
        if name == "net_-1/fc2_weights" or name == "net_-1/fc2_biases":
            transfer_params.remove(param)
        elif name == "net_-1/fc3_weights" or name == "net_-1/fc3_biases":
            transfer_params.remove(param)

    with network.graph.as_default():
        transfer_saver = tf.train.Saver(transfer_params)
    transfer_saver.save(
        sess, str(model_folder / 'transfer_model/nofc2'
                  / '{}_transfer_params'.format(GYM_ENV_NAME)))

    # Remove fc1 weights
    for param in transfer_params[:]:
        name = param.op.name
        if name == "net_-1/fc1_weights" or name == "net_-1/fc1_biases":
            transfer_params.remove(param)

    with network.graph.as_default():
        transfer_saver = tf.train.Saver(transfer_params)
    transfer_saver.save(
        sess, str(model_folder / 'transfer_model/nofc1'
                  / '{}_transfer_params'.format(GYM_ENV_NAME)))

    # Remove conv3 weights
    if args.use_mnih_2015:
        for param in transfer_params[:]:
            name = param.op.name
            if name == "net_-1/conv3_weights" or name == "net_-1/conv3_biases":
                transfer_params.remove(param)

        with network.graph.as_default():
            transfer_saver = tf.train.Saver(transfer_params)
        transfer_saver.save(
            sess, str(model_folder / 'transfer_model/noconv3'
                      / '{}_transfer_params'.format(GYM_ENV_NAME)))

    # Remove conv2 weights
    for param in transfer_params[:]:
        name = param.op.name
        if name == "net_-1/conv2_weights" or name == "net_-1/conv2_biases":
            transfer_params.remove(param)

    with network.graph.as_default():
        transfer_saver = tf.train.Saver(transfer_params)
    transfer_saver.save(
        sess, str(model_folder / 'transfer_model/noconv2'
                  / '{}_transfer_params'.format(GYM_ENV_NAME)))

    max_output_value_file = model_folder / 'transfer_model/max_output_value'
    with max_output_value_file.open('w') as f:
        f.write(str(classify_demo.max_val))
    logger.info('Data saved!')

    sess.close()
