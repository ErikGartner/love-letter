# -*- coding: utf-8 -*-

"""
Abstract Agent for a Love Letter AI
"""

import random

from loveletter.card import Card
from loveletter.player import PlayerAction, PlayerActionTools

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
        player_self = game.player_turn()
        opponents = game.opponent_turn()

        possible_actions = [
            # Actions always targeting the current player
            PlayerAction(Card.king, player_self, Card.noCard, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.guard, player_self, Card.noCard, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.priest, player_self, Card.noCard, Card.noCard, Card.noCard, player_self, 0),
            PlayerAction(Card.baron, player_self, Card.noCard, Card.noCard, Card.noCard, player_self, 0),

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

        actions = [action for action in possible_actions
                   if game.is_action_valid(action)]
        return actions
