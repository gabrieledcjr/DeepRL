#!/usr/bin/env python3
import numpy as np
import logging
import os

from common.replay_memory import ReplayMemory
from common.game_state import GameState

logger = logging.getLogger("dqn")

def run_dqn(args):
    """
    Baseline:
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --cuda-devices=0 --optimizer=Adam --lr=0.0001 --decay=0.0 --momentum=0.0 --epsilon=0.001 --gpu-fraction=0.222
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --cuda-devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.00001 --gpu-fraction=0.222

    Transfer with Human Memory:
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --cuda-devices=0 --optimizer=Adam --lr=0.0001 --decay=0.0 --momentum=0.0 --epsilon=0.001 --observe=0 --use-transfer --load-memory
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --cuda-devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.00001 --observe=0 --use-transfer --load-memory
    python3 run_experiment.py breakout --cuda-devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.01 --observe=0 --use-transfer --load-memory --train-max-steps=20500000

    Transfer with Human Advice and Human Memory:
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --cuda-devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.00001 --observe=0 --use-transfer --load-memory --use-human-model-as-advice --advice-confidence=0. --psi=0.9999975 --train-max-steps=20500000

    Human Advice only with Human Memory:
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --cuda-devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.00001 --observe=0 --load-memory --use-human-model-as-advice --advice-confidence=0.75 --psi=0.9999975
    """
    from dqn_net import DqnNet
    from dqn_net_class import DqnNetClass
    from dqn_training import DQNTraining
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        if args.cuda_devices != '':
            os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    import tensorflow as tf

    if not os.path.exists('results/dqn'):
        os.makedirs('results/dqn')

    if args.folder is not None:
        folder = 'results/dqn/{}_{}'.format(args.gym_env.replace('-', '_'), args.folder)
    else:
        folder = 'results/dqn/{}_{}'.format(args.gym_env.replace('-', '_'), args.optimizer.lower())
        end_str = ''

        if args.unclipped_reward:
            end_str += '_rawreward'
        elif args.log_scale_reward:
            end_str += '_logreward'
        if args.transformed_bellman:
            end_str += '_transformedbell'
        if args.target_consistency:
            end_str += '_tcloss'

        if args.use_transfer:
            end_str += '_transfer'
            if args.not_transfer_conv2:
                end_str += '_noconv2'
            elif args.not_transfer_conv3 and args.use_mnih_2015:
                end_str += '_noconv3'
            elif args.not_transfer_fc1:
                end_str += '_nofc1'
            elif args.not_transfer_fc2:
                end_str += '_nofc2'

        if args.use_human_model_as_advice:
            end_str += '_modelasadvice'

        folder += end_str

    if args.append_experiment_num is not None:
        folder += '_' + args.append_experiment_num

    if args.cpu_only:
        device = '/cpu:0'
        gpu_options = None
    else:
        device = '/gpu:'+os.environ["CUDA_VISIBLE_DEVICES"]
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False)

    game_state = GameState(env_id=args.gym_env, display=False, no_op_max=30, human_demo=False, episode_life=True)
    human_net = None
    sess_human = None
    if args.use_human_model_as_advice:
        if args.advice_folder is not None:
            advice_folder = args.advice_folder
        else:
            advice_folder = "{}_networks_classifier_{}".format(args.gym_env.replace('-', '_'), "adam")
        DqnNetClass.use_gpu = not args.cpu_only
        human_net = DqnNetClass(
            args.resized_height, args.resized_width,
            args.phi_len, game_state.env.action_space.n, args.gym_env,
            optimizer="Adam", learning_rate=0.0001, epsilon=0.001,
            decay=0., momentum=0., folder=advice_folder, device='/cpu:0')
        sess_human = tf.Session(config=config, graph=human_net.graph)
        human_net.initializer(sess_human)
        human_net.load()

    # prepare session
    sess = tf.Session(config=config)

    replay_memory = ReplayMemory(
        args.resized_width, args.resized_height,
        np.random.RandomState(),
        max_steps=args.replay_memory,
        phi_length=args.phi_len,
        num_actions=game_state.env.action_space.n,
        wrap_memory=True,
        full_state_size=game_state.clone_full_state().shape[0])

    # baseline learning
    if not args.use_transfer:
        DqnNet.use_gpu = not args.cpu_only
        net = DqnNet(
            sess, args.resized_height, args.resized_width, args.phi_len,
            game_state.env.action_space.n, args.gym_env, gamma=args.gamma,
            optimizer=args.optimizer, learning_rate=args.lr,
            epsilon=args.epsilon, decay=args.decay, momentum=args.momentum,
            verbose=args.verbose, folder=folder,
            slow=args.use_slow, tau=args.tau, device=device,
            transformed_bellman=args.transformed_bellman,
            target_consistency_loss=args.target_consistency)

    # transfer using existing model
    else:
        if args.transfer_folder is not None:
            transfer_folder = args.transfer_folder
        else:
            transfer_folder = 'results/pretrain_models/{}'.format(args.gym_env.replace('-', '_'))
            end_str = ''
            end_str += '_mnih2015'
            end_str += '_l2beta1E-04_batchprop'  #TODO: make this an argument
            transfer_folder += end_str
            transfer_folder += '/transfer_model'

        DqnNet.use_gpu = not args.cpu_only
        net = DqnNet(
            sess, args.resized_height, args.resized_width, args.phi_len,
            game_state.env.action_space.n, args.gym_env, gamma=args.gamma,
            optimizer=args.optimizer, learning_rate=args.lr,
            epsilon=args.epsilon, decay=args.decay, momentum=args.momentum,
            verbose=args.verbose, folder=folder,
            slow=args.use_slow, tau=args.tau,
            transfer=True, transfer_folder=transfer_folder,
            not_transfer_conv2=args.not_transfer_conv2,
            not_transfer_conv3=args.not_transfer_conv3,
            not_transfer_fc1=args.not_transfer_fc1,
            not_transfer_fc2=args.not_transfer_fc2, device=device,
            transformed_bellman=args.transformed_bellman,
            target_consistency_loss=args.target_consistency)

    demo_memory_folder = None
    if args.load_memory:
        if args.demo_memory_folder is not None:
            demo_memory_folder = args.demo_memory_folder
        else:
            demo_memory_folder = "{}_demo_samples".format(args.gym_env.replace('-', '_'))

    if args.unclipped_reward:
        reward_type = ''
    elif args.log_scale_reward:
        reward_type = 'LOG'
    else:
        reward_type = 'CLIP'

    experiment = DQNTraining(
        sess, net, game_state, args.resized_height, args.resized_width,
        args.phi_len, args.batch, args.gym_env,
        args.gamma, args.observe, args.explore, args.final_epsilon,
        args.init_epsilon, replay_memory,
        args.update_freq, args.save_freq, args.eval_freq,
        args.eval_max_steps, args.c_freq,
        folder, load_demo_memory=args.load_memory,
        demo_memory_folder=demo_memory_folder,
        train_max_steps=args.train_max_steps,
        human_net=human_net, confidence=args.advice_confidence, psi=args.psi,
        train_with_demo_steps=args.train_with_demo_steps,
        use_transfer=args.use_transfer, reward_type=reward_type)
    experiment.run()

    if args.use_human_model_as_advice:
        sess_human.close()

    sess.close()
