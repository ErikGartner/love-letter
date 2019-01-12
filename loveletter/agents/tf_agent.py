"""Agent with uses A3C trained network"""

import random

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from loveletter.env import LoveLetterEnv
from loveletter.agents.random import AgentRandom
from loveletter.agents.agent import Agent
from loveletter.trainers.a3c_model import ActorCritic



class TFAgent(Agent):
    '''Agent which leverages tensorflow'''

    def __init__(self,
                 model_path,
                 seed=451):
        self._seed = seed
        self._idx = 0
        self.env = LoveLetterEnv(AgentRandom(seed), seed)
        self.vec_env = DummyVecEnv([lambda: self.env])  # The algorithms require a vectorized environment to run

        state = self.env.reset()

        self._model = PPO2(MlpPolicy, self.vec_env, verbose=0, tensorboard_log="./tensorboard/")
        self._model.load(model_path)

    def _move(self, game):
        '''Return move which ends in score hole'''
        assert game.active()
        self._idx += 1

        state = self.env.force(game)
        action_idx = self._model.predict(state, deterministic=True)[0]

        player_action = self.env.action_from_index(action_idx, game)
        if player_action is None:
            # print("ouch")
            options = Agent.valid_actions(game, self._seed + self._idx)
            if len(options) < 1:
                raise Exception("Unable to play without actions")

            random.seed(self._seed + self._idx)
            return random.choice(options)

        # print("playing ", self._idx, player_action)
        return player_action
