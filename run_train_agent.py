"""Kick off for training A3C agent training"""

import argparse
import datetime

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from loveletter.env import LoveLetterEnv
from loveletter.arena import Arena
from loveletter.agents.random import AgentRandom
from loveletter.agents.tf_agent import TFAgent
from loveletter.agents.agent import Agent

from loveletter.trainers.a3c_model import ActorCritic
from loveletter.trainers.a3c_train import train
from loveletter.trainers.a3c_test import test


parser = argparse.ArgumentParser(description='RL for Love Letter')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps for update (default: 20)')
parser.add_argument('--total-steps', type=int, default=1e6, metavar='NS',
                    help='number of total steps (default: 1M)')
parser.add_argument('--save-name', metavar='FN', default='default_model',
                    help='path/prefix for the filename to save shared model\'s parameters')
parser.add_argument('--load-name', default=None, metavar='SN',
                    help='path/prefix for the filename to load shared model\'s parameters')
parser.add_argument('--log-dir', default="./tensorboard/", metavar='path',
                    help='path to the tensorboard log directory')


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[512, dict(pi=[256, 128],
                                                               vf=[256, 128])],
                                           feature_extraction="mlp")


if __name__ == '__main__':
    args = parser.parse_args()

    if args.load_name:
        env = SubprocVecEnv([lambda: LoveLetterEnv(TFAgent(args.load_name, args.seed + i))
                             for i in range(args.num_processes)])
    else:
        env = SubprocVecEnv([lambda: LoveLetterEnv(AgentRandom(args.seed + i))
                             for i in range(args.num_processes)])

    model = PPO2(CustomPolicy,
                 env,
                 verbose=0,
                 tensorboard_log=args.log_dir,
                 learning_rate=args.lr,
                 n_steps=args.num_steps,
                 nminibatches=5,
                 policy_kwargs=policy_kwargs)

    if args.load_name:
        model.load(args.load_name)

    model.learn(total_timesteps=int(args.total_steps),
                callback=None,
                tb_log_name='PPO2 %s' % datetime.datetime.now().strftime('%H-%M-%S'))
    model.save(args.save_name)
