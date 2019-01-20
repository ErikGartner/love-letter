from operator import itemgetter

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from .game import Game
from .card import Card
from .player import PlayerAction, PlayerTools
from .agents.random import AgentRandom


NBR_PLAYERS = 4
NBR_ACTIONS = 11 * NBR_PLAYERS + NBR_PLAYERS
SPACE_SIZE = 1344 + NBR_ACTIONS

REWARD_WIN = 1
REWARD_LOSE = -1
REWARD_INVALID_ACTION = -1.50


class LoveLetterEnv(gym.Env):
    """Love Letter Game Environment"""

    def __init__(self, agent_other, seed=451):

        self.action_space = spaces.Discrete(SPACE_SIZE)
        self.observation_space = spaces.Box(low=0, high=1, shape=(SPACE_SIZE,),
                                            dtype=np.float32)

        self._agent_other = AgentRandom(
            seed) if agent_other is None else agent_other
        self.seed(seed)
        self.reset()
        self._game = Game.new(4, self.np_random.random_integers(5000000))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        player_action = self.action_from_index(action)

        if player_action is None or not self._game.is_action_valid(player_action):
            return self._state(), REWARD_INVALID_ACTION, True, {"round": self._game.round()}

        self._game, reward = LoveLetterEnv.advance_game(
            self._game, player_action, self._agent_other)

        done = self._game.over() or not PlayerTools.is_playing(
            self._game.players()[0])

        return self._state(), reward, done, {"round": self._game.round()}

    def reset(self):
        self._game = Game.new(4, self.np_random.random_integers(5000000))
        return self._state()

    def force(self, game):
        """Force the environment to a certain game state"""
        self._game = game
        return game.state()

    def _state(self, game=None):
        """
        Gets the current state from game with additional state information.
        """
        game = self._game if game is None else game

        action_candidates = self.actions_set(game)
        actions = [game.is_action_valid(a) for a in self.actions_set(game)]
        actions = np.array(actions, dtype=np.int8)
        return np.concatenate([actions, game.state()])

    @staticmethod
    def advance_game(game, action, agent):
        """Advance a game with an action

        * Play an action
        * Advance the game using the agent
        * Return the game pending for the same player turn _unless_ the game ends

        returns <game, reward>
        """
        if not game.is_action_valid(action):
            return game, REWARD_INVALID_ACTION

        player_idx = game.player_turn()
        game_current, _ = game.move(action)
        while game_current.active():
            if not game_current.is_current_player_playing():
                game_current = game_current.skip_eliminated_player()
            elif game_current.player_turn() != player_idx:
                game_current, _ = game_current.move(agent.move(game_current))
            else:
                break

        # print("Round", game.round(), '->', game_current.round(), ':', 'OVER' if game_current.over() else 'RUNN')

        if game_current.over():
            if game_current.winner() == player_idx:
                return game_current, REWARD_WIN
            else:
                return game_current, REWARD_LOSE

        return game_current, 0


    def action_by_score(self, scores, game=None):
        """
        Returns best action based on assigned scores

        return (action, score, idx)
        """
        if len(scores) != NBR_ACTIONS:
            raise Exception("Invalid scores length: {}".format(len(scores)))
        game = self._game if game is None else game

        assert game.active()
        actions_possible = self.actions_set(game)

        actions = [(action, score, idx) for action, score, idx in
                   zip(actions_possible,
                       scores,
                       range(len(actions_possible)))
                   if game.is_action_valid(action)]

        action = max(actions, key=itemgetter(2))
        return action

    def action_from_index(self, action_index, game=None):
        """Returns valid action based on index and game"""
        game = self._game if game is None else game

        action_candidates = self.actions_set(game)

        actions = [(idx, action) for idx, action in
                   enumerate(action_candidates)
                   if game.is_action_valid(action) and idx == action_index]

        return actions[0][1] if len(actions) == 1 else None

    def actions_possible(self, game=None):
        """Returns valid (idx, actions) based on a current game"""
        game = self._game if game is None else game

        action_candidates = self.actions_set(game)

        actions = [(idx, action) for idx, action in
                   enumerate(action_candidates)
                   if game.is_action_valid(action)]

        return actions

    def actions_set(self, game=None):
        """Returns all actions for a game"""
        game = self._game if game is None else game

        player_self = game.player_turn()
        opponents = game.opponent_turn()

        possible_actions = [
            # Actions always targeting the current player
            PlayerAction(Card.prince, player_self, Card.noCard, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.handmaid, player_self, Card.noCard, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.countess, player_self, Card.noCard, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.princess, player_self, Card.noCard, Card.noCard, Card.noCard, player_self, 0)
        ]
        for rel_idx in range(len(game._players)):
            target_idx = game.absolute_player_idx(rel_idx, player_self)

            possible_actions.extend([
                PlayerAction(Card.guard,
                             target_idx,
                             Card.priest,
                             Card.noCard, Card.noCard, player_self, 0),
                PlayerAction(Card.guard,
                             target_idx,
                             Card.baron,
                             Card.noCard, Card.noCard, player_self, 0),
                PlayerAction(Card.guard,
                             target_idx,
                             Card.handmaid,
                             Card.noCard, Card.noCard, player_self, 0),
                PlayerAction(Card.guard,
                             target_idx,
                             Card.prince,
                             Card.noCard, Card.noCard, player_self, 0),
                PlayerAction(Card.guard,
                             target_idx,
                             Card.king,
                             Card.noCard, Card.noCard, player_self, 0),
                PlayerAction(Card.guard,
                             target_idx,
                             Card.countess,
                             Card.noCard, Card.noCard, player_self, 0),
                PlayerAction(Card.guard,
                             target_idx,
                             Card.princess,
                             Card.noCard, Card.noCard, player_self, 0),
                PlayerAction(Card.priest,
                             target_idx,
                             Card.noCard,
                             Card.noCard, Card.noCard, player_self, 0),
                PlayerAction(Card.baron,
                             target_idx,
                             Card.noCard,
                             Card.noCard, Card.noCard, player_self, 0),
                PlayerAction(Card.king,
                             target_idx,
                             Card.noCard,
                             Card.noCard, Card.noCard, player_self, 0),
                PlayerAction(Card.prince,
                             target_idx,
                             Card.noCard,
                             Card.noCard, Card.noCard, player_self, 0)
                ])

        return possible_actions
