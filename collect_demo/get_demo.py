#!/usr/bin/env python3
import argparse
import coloredlogs, logging

from common.game_state import GameState
from common.util import prepare_dir, LogFormatter
from collect_demo import CollectDemonstration

logger = logging.getLogger('get_demo')
coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s')

def get_demo(args):
    """
    Requirements: sudo apt-get install python3-tk
    python3 get_demo.py --gym-env=PongNoFrameskip-v4 --num-episodes=5 --demo-time-limit=5
    """
    if args.demo_memory_folder is not None:
        demo_memory_folder = 'demo_samples/{}'.format(args.demo_memory_folder)
    else:
        demo_memory_folder = 'demo_samples/{}'.format(args.gym_env.replace('-', '_'))

    if args.append_experiment_num is not None:
        demo_memory_folder += '_' + args.append_experiment_num

    prepare_dir(demo_memory_folder, empty=True)
    fh = logging.FileHandler('{}/collect.log'.format(demo_memory_folder), mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = LogFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    game_state = GameState(env_id=args.gym_env, display=True, human_demo=True)
    collect_demo = CollectDemonstration(
        game_state,
        84, 84, 4,
        args.gym_env,
        folder=demo_memory_folder,
        create_gif=args.create_gif)
    collect_demo.run_episodes(
        args.num_episodes,
        minutes_limit=args.demo_time_limit,
        demo_type=0)
    game_state.close()

def main():
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()

    parser.add_argument('--gym-env', type=str, default='PongNoFrameskip-v4', help='OpenAi Gym environment ID')
    parser.add_argument('--append-experiment-num', type=str, default=None)
    parser.add_argument('--num-episodes', type=int, default=5, help='number of episodes')
    parser.add_argument('--demo-time-limit', type=int, default=5, help='time limit per episode')
    parser.add_argument('--create-gif', action='store_true')
    parser.set_defaults(create_gif=False)
    parser.add_argument('--demo-memory-folder', type=str, default=None)

    args = parser.parse_args()

    logger.info('Collecting demonstration...')
    get_demo(args)


if __name__ == "__main__":
    main()
