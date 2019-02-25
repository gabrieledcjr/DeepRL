#!/usr/bin/env python
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle

parser = argparse.ArgumentParser()
parser.add_argument('env', type=str)
parser.add_argument('--deep-rl-algo', type=str, default='dqn')
args = parser.parse_args()

game = args.env
deep_rl_algo = args.deep_rl_algo
game_env_type = 'NoFrameskip_v4'
location = 'lower right'
ncol = 2
last_base = []
num_data = [1, 2, 3, 4]

if game == 'Asterix':
    pass
elif game == 'Breakout':
    pass
elif game == 'Freeway':
    location = "upper left"
elif game == 'Gopher':
    location = "upper left"
    ncol=1
elif game == 'MsPacman':
    pass
elif game == 'NameThisGame':
    pass
elif game == 'Pong':
    pass
elif game == 'SpaceInvaders':
    pass
else:
    print("Invalid game!!!")
    sys.exit()

gym_env = game + game_env_type
print(gym_env)

#sns.set_style("darkgrid")
sns.set(context='paper', style='darkgrid', rc={'figure.figsize':(7,5)})
LW = 1.5
ALPHA = 0.1
MARKERSIZE = 12

N = 1

if deep_rl_algo == 'dqn':
    MAX_TIMESTEPS = 30000000
elif deep_rl_algo == 'a3c':
    MAX_TIMESTEPS = 50000000
else:
    print("Invalid algorithm!!!")
    sys.exit()

PER_EPOCH = 1000000

def create_dataframe(rewards_all_trials, per_epoch=0, max_timesteps=0):
    if per_epoch == 0:
        per_epoch = PER_EPOCH
    if max_timesteps == 0:
        max_timesteps = MAX_TIMESTEPS
    timestep = [ t/per_epoch for t in range(0, (max_timesteps+per_epoch), per_epoch) ]
    print(timestep)
    d = {}

    for data_idx, reward_data in enumerate(rewards_all_trials):
        rewards = []
        for r in sorted(reward_data['eval'].keys()):
            #print(r)
            if r <= max_timesteps and r % per_epoch == 0:
                #print(reward_data['eval'][r])
                rewards.append(reward_data['eval'][r][0]) ### <=== (reward, steps, num_episodes)
        d['Rewards{}'.format(data_idx+1)] = rewards
        print(len(rewards))
    df = pd.DataFrame(data=d, index=timestep)
    df.index.name = 'Timestep'
    df = df.iloc[::N]
    return df


plt.figure()

def plot(
    ex_type='', color='green',
    marker='o', markersize=MARKERSIZE,
    lw=LW, linestyle=None, label='test', num_data=[1,2,3,4], last=None):
    ''' creates dataframe and plots graph '''
    rewards_all_trials = []
    for data_idx in num_data:
        print ('results_kamiak/{}/'.format(deep_rl_algo) + gym_env + '{}_{}/'.format(ex_type, data_idx) + gym_env + '-{}-rewards.pkl'.format(deep_rl_algo))
        r_data = pickle.load(open('results_kamiak/{}/'.format(deep_rl_algo) + gym_env + '{}_{}/'.format(ex_type, data_idx) + gym_env + '-{}-rewards.pkl'.format(deep_rl_algo), 'rb'))
        if deep_rl_algo == 'dqn':
            print(sorted(r_data['eval'].keys()))
            if r_data['eval'].get(30000000) is None:
                print("GOT HERE")
                if last is None:
                    r_data['eval'][30000000] = r_data['eval'][29000000] if deep_rl_algo == 'a3c' else r_data['eval'][28000000]
                else:
                    r_data['eval'][30000000] = last[data_idx-1]
        rewards_all_trials.append(r_data)

    df_rewards = create_dataframe(rewards_all_trials)
    while len(rewards_all_trials):
        del rewards_all_trials[0]

    df_rewards_mean = df_rewards.mean(axis=1)
    print (df_rewards.mean(axis=0))
    df_rewards_std = df_rewards.std(axis=1)

    plt.plot(
        df_rewards_mean.index,
        df_rewards_mean,
        color=sns.xkcd_rgb[color],
        marker=marker,
        markersize=markersize,
        markevery=3,
        lw=lw,
        linestyle=linestyle,
        label=label)
    plt.fill_between(
        df_rewards_std.index,
        df_rewards_mean - df_rewards_std,
        df_rewards_mean + df_rewards_std,
        color=sns.xkcd_rgb[color],
        alpha=ALPHA)

# plt.plot(
#     df_rewards_transfer_then_pretrain_indqn_rms_mean.index,
#     df_rewards_transfer_then_pretrain_indqn_rms_mean,
#     color=sns.xkcd_rgb["ocean blue"],
#     marker='d',
#     markersize=MARKERSIZE,
#     lw=LW,
#     label='pretrained model+pretrained DQN')
# plt.fill_between(
#     df_rewards_transfer_then_pretrain_indqn_rms_std.index,
#     df_rewards_transfer_then_pretrain_indqn_rms_mean - 2*df_rewards_transfer_then_pretrain_indqn_rms_std,
#     df_rewards_transfer_then_pretrain_indqn_rms_mean + 2*df_rewards_transfer_then_pretrain_indqn_rms_std,
#     color=sns.xkcd_rgb["ocean blue"],
#     alpha=ALPHA)
#
# plt.plot(
#     df_rewards_transfer_then_pretrain_indqn_noexplore_rms_mean.index,
#     df_rewards_transfer_then_pretrain_indqn_noexplore_rms_mean,
#     color=sns.xkcd_rgb["orange"],
#     marker='o',
#     markersize=MARKERSIZE,
#     lw=LW,
#     label='pretrained model+pretrained DQN (no explore)')
# plt.fill_between(
#     df_rewards_transfer_then_pretrain_indqn_noexplore_rms_std.index,
#     df_rewards_transfer_then_pretrain_indqn_noexplore_rms_mean - 2*df_rewards_transfer_then_pretrain_indqn_noexplore_rms_std,
#     df_rewards_transfer_then_pretrain_indqn_noexplore_rms_mean + 2*df_rewards_transfer_then_pretrain_indqn_noexplore_rms_std,
#     color=sns.xkcd_rgb["orange"],
#     alpha=ALPHA)

# baseline
ex_type = '_rms'
if deep_rl_algo == 'a3c':
    ex_type = '_mnih2015'
#if game == 'newbreakout':
#    ex_type += '_autostart'
plot(
    ex_type=ex_type,
    color='dark grey',
    marker=None,
    markersize=0,
    lw=LW+1,
    linestyle='--',
    label=deep_rl_algo.upper(),
    num_data=num_data,
    last=last_base)

ex_type = '_mnih2015_rawreward_transformedbell_transfer_noconv2'
plot(
    ex_type=ex_type,
    color='violet',
    marker='X',
    label='PMfA3C-TB conv1',
    num_data=num_data)

ex_type = '_mnih2015_rawreward_transformedbell_transfer_noconv3'
plot(
    ex_type=ex_type,
    color='cobalt',
    marker='o',
    label='PMfA3C-TB conv2',
    num_data=num_data)

ex_type = '_mnih2015_rawreward_transformedbell_transfer_nofc1'
plot(
    ex_type=ex_type,
    color='blue green',
    marker='d',
    label='PMfA3C-TB conv3',
    num_data=num_data)

ex_type = '_mnih2015_rawreward_transformedbell_transfer_nofc2'
plot(
    ex_type=ex_type,
    color='lavender blue',
    marker='^',
    label='PMfA3C-TB fc1',
    num_data=num_data)

ex_type = '_mnih2015_rawreward_transformedbell_transfer'
plot(
    ex_type=ex_type,
    color='reddish',
    marker='P',
    label='PMfA3C-TB',
    num_data=num_data)


# # MTL pretrained model with 1 demo
# if game in ['pong', 'beamrider']:
#    plot(
#        ex_type='_transfer_nofc2_use_mnih_use_lstm_full_newvloss_from_mtl_exclude4demoeps',
#        color='dark blue',
#        marker='>',
#        label='PMfA3C (1-vs-all, 1-demo)')

plt.xlabel('Steps (in millions)', fontsize='x-large')
plt.ylabel('Reward', fontsize='x-large')
plt.legend(loc=location, fontsize='large', ncol=ncol)#, prop={'size':9})

header = game
plt.title('{}'.format(game), fontsize='x-large')
plt.savefig('plots/{}_{}_transfer.pdf'.format(game, deep_rl_algo), bbox_inches='tight')
plt.savefig('plots/{}_{}_transfer.png'.format(game, deep_rl_algo), bbox_inches='tight')
plt.show()
