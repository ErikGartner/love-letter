# -*- coding: utf-8 -*-

"""
Abstract Agent for a Love Letter AI
"""

import random

from loveletter.card import Card
from loveletter.player import PlayerAction

class Agent():
    """Abstract Class for agent to play Love Letter."""


    def move(self, game):
        """Return a Player Action based on a game state"""
        return self._move(game)

    def _move(self, game):
        """Return a Player Action based on a game state"""
        raise NotImplementedError("Class {} doesn't implement _move()".format(
            self.__class__.__name__))

    @staticmethod
    def valid_actions(game, seed=451):
        """Returns valid moves based on a current game"""
        random.seed(seed + game.round())
        player_self = game.player_turn()
        opponents = game.opponent_turn()

        actions_possible = [
            PlayerAction(Card.guard, random.choice(opponents), Card.priest, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.guard, random.choice(opponents), Card.baron, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.guard, random.choice(opponents), Card.handmaid, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.guard, random.choice(opponents), Card.prince, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.guard, random.choice(opponents), Card.king, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.guard, random.choice(opponents), Card.countess, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.guard, random.choice(opponents), Card.princess, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.priest, random.choice(opponents), Card.noCard, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.baron, random.choice(opponents), Card.noCard, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.king, random.choice(opponents), Card.noCard, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.prince, random.choice(opponents), Card.noCard, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.prince, player_self, Card.noCard, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.handmaid, player_self, Card.noCard, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.countess, player_self, Card.noCard, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.princess, player_self, Card.noCard, Card.noCard, Card.noCard, player_self, 0)
        ]
        actions = [action for action in actions_possible if game.is_action_valid(action)]

        return actions
