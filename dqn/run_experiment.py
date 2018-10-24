#!/usr/bin/env python3
import argparse
import multiprocessing
import os
import numpy as np
import coloredlogs, logging

from time import sleep
from dqn import run_dqn
from classify_demo import classify_demo
from common.game_state import GameState
from common.replay_memory import ReplayMemory

logger = logging.getLogger()
coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s')

def get_demo(args):
    """
    Human:
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --demo-time-limit=5 --collect-demo --demo-type=0 --file-num=1

    Random:
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --demo-time-limit=5 --collect-demo --demo-type=1 --file-num=1

    Model:
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --demo-time-limit=5 --collect-demo --demo-type=2 --file-num=1
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --demo-time-limit=5 --collect-demo --demo-type=2 --model-folder=pong_networks_rms_1 --file-num=1
    """
    if args.demo_type == 2:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import tensorflow as tf
        from dqn_net import DqnNet
    from collect_demo import CollectDemonstration

    if args.folder is not None:
        folder = '{}_{}'.format(args.gym_env.replace('-', '_'), args.folder)
    else:
        folder = '{}_demo_samples'.format(args.gym_env.replace('-', '_'))
        if args.demo_type == 1:
            folder = '{}_demo_samples_random'.format(args.gym_env.replace('-', '_'))
        elif args.demo_type == 2:
            folder = '{}_demo_samples_model'.format(args.gym_env.replace('-', '_'))

    game_state = GameState(env_id=args.gym_env, display=False, no_op_max=30, human_demo=False, episode_life=True)

    replay_memory = ReplayMemory(
        args.resized_width, args.resized_height,
        np.random.RandomState(),
        max_steps=args.demo_time_limit * 5000,
        phi_length=args.phi_len,
        num_actions=game_state.env.action_space.n,
        wrap_memory=True,
        full_state_size=game_state.clone_full_state().shape[0],
        clip_reward=True)

    model_net = None
    if args.demo_type == 2: # From model
        if args.model_folder is not None:
            model_folder = args.model_folder
        else:
            model_folder = '{}_networks_{}'.format(args.gym_env.replace('-', '_'), args.optimizer.lower())
        sess = tf.Session()
        with tf.device('/cpu:0'):
            model_net = DqnNet(
                sess, args.resized_height, args.resized_width, args.phi_len,
                game_state.env.action_space.n, args.gym_env, gamma=args.gamma, copy_interval=args.c_freq,
                optimizer=args.optimizer, learning_rate=args.lr,
                epsilon=args.epsilon, decay=args.decay, momentum=args.momentum,
                verbose=args.verbose, path=None, folder=None,
                slow=args.use_slow, tau=args.tau)
            model_net.load(folder=model_folder)

    collect_demo = CollectDemonstration(
        game_state, args.resized_height, args.resized_width, args.phi_len,
        args.gym_env, replay_memory, terminate_loss_of_life=args.terminate_life_loss,
        folder=folder, sample_num=args.file_num
    )
    collect_demo.run(
        minutes_limit=args.demo_time_limit,
        demo_type=args.demo_type,
        model_net=model_net)

def main():
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()

    parser.add_argument('--gym-env', type=str, default='PongNoFrameskip-v4', help='OpenAi Gym environment ID')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--observe', type=int, default=50000)
    parser.add_argument('--explore', type=int, default=1000000)
    parser.add_argument('--final-epsilon', type=float, default=0.1)
    parser.add_argument('--init-epsilon', type=float, default=1.0)
    parser.add_argument('--replay-memory', type=int, default=1000000)
    parser.add_argument('--resized-width', type=int, default=84)
    parser.add_argument('--resized-height', type=int, default=84)
    parser.add_argument('--phi-len', type=int, default=4)
    parser.add_argument('--batch', type=int, default=32)

    parser.add_argument('--update-freq', type=int, default=4)
    parser.add_argument('--save-freq', type=int, default=1000000)
    parser.add_argument('--eval-freq', type=int, default=1000000)
    parser.add_argument('--eval-max-steps', type=int, default=125000)
    parser.add_argument('--train-max-steps', type=int, default=41 * 10**6)

    parser.add_argument('--c-freq', type=int, default=10000)
    parser.add_argument('--use-slow', action='store_true')
    parser.set_defaults(use_slow=False)
    parser.add_argument('--tau', type=float, default=1.)

    parser.add_argument('--optimizer', type=str, default='RMS')
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--decay', type=float, default=0.95)
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--epsilon', type=float, default=0.00001)

    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--append-experiment-num', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.set_defaults(verbose=False)

    parser.add_argument('--cuda-devices', type=str, default='')
    parser.add_argument('--gpu-fraction', type=float, default=0.333)
    parser.add_argument('--cpu-only', action='store_true')
    parser.set_defaults(cpu_only=False)

    parser.add_argument('--use-transfer', action='store_true')
    parser.set_defaults(use_transfer=False)
    parser.add_argument('--transfer-folder', type=str, default=None)
    parser.add_argument('--not-transfer-conv2', action='store_true')
    parser.set_defaults(not_transfer_conv2=False)
    parser.add_argument('--not-transfer-conv3', action='store_true')
    parser.set_defaults(not_transfer_conv3=False)
    parser.add_argument('--not-transfer-fc1', action='store_true')
    parser.set_defaults(not_transfer_fc1=False)
    parser.add_argument('--not-transfer-fc2', action='store_true')
    parser.set_defaults(not_transfer_fc2=False)

    parser.add_argument('--use-human-model-as-advice', action='store_true')
    parser.set_defaults(use_human_model_as_advice=False)
    parser.add_argument('--advice-confidence', type=float, default=0.)
    parser.add_argument('--advice-folder', type=str, default=None)
    parser.add_argument('--psi', type=float, default=0.)

    parser.add_argument('--load-memory', action='store_true')
    parser.set_defaults(load_memory=False)
    parser.add_argument('--demo-memory-folder', type=str, default=None)

    parser.add_argument('--train-with-demo-steps', type=int, default=0)

    parser.add_argument('--collect-demo', action='store_true')
    parser.set_defaults(collect_demo=False)
    parser.add_argument('--demo-type', type=int, default=0, help='[0] human, [1] random, [2] model')
    parser.add_argument('-n', '--file-num', type=int, default=1)
    parser.add_argument('--model-folder', type=str, default=None)
    parser.add_argument('--demo-time-limit', type=int, default=5) # 5 minutes
    parser.add_argument('--terminate-life-loss', action='store_true')
    parser.set_defaults(terminate_life_loss=False)

    parser.add_argument('--classify-demo', action='store_true')
    parser.set_defaults(classify_demo=False)

    # Alternatives to reward clipping
    parser.add_argument('--unclipped-reward', action='store_true', help='use raw reward')
    parser.set_defaults(unclipped_reward=False)
    # DQfD Hester, et. al 2017
    parser.add_argument('--log-scale-reward', action='store_true', help='use log scale reward r = sign(r) * log(1 + abs(r)) from DQfD (Hester et. al)')
    parser.set_defaults(log_scale_reward=False)
    # Ape-X Pohlen, et. al 2018
    parser.add_argument('--transformed-bellman', action='store_true', help='use transofrmed bellman equation')
    parser.set_defaults(transformed_bellman=False)
    parser.add_argument('--target-consistency', action='store_true', help='use target consistency (TC) loss')
    parser.set_defaults(target_consistency=False)

    args = parser.parse_args()

    if args.collect_demo:
        logger.info('Collecting demonstration...')
        sleep(2)
        get_demo(args)
    elif args.classify_demo:
        logger.info('Classifying human demonstration...')
        sleep(2)
        classify_demo(args)
    else:
        logger.info('Running DQN...')
        sleep(2)
        run_dqn(args)


if __name__ == "__main__":
    main()
