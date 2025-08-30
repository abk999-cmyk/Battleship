"""Battleship web game logic package."""

from .ship import Ship
from .board import Board
from .player import Player, HumanPlayer, AIPlayer
from .game import BattleshipGame

# Advanced AI Agents
from .AI_agent import AIPlayer as AIPlayerClassic
from .AI_agent2 import AIPlayer2
from .AI_agent3 import AIAgent3
from .AI_agent4 import AIAgent4

# Testing Agents
from .AI_testing_agents import (
    NaiveAgent1, NaiveAgent2, NaiveAgent3, NaiveAgent4, NaiveAgent5,
    NaiveAgent6, NaiveAgent7, NaiveAgent8, NaiveAgent9, NaiveAgent10,
    UltimateBattleshipAgent
)

__all__ = [
    'Ship', 'Board', 'Player', 'HumanPlayer', 'AIPlayer', 'BattleshipGame',
    'AIPlayerClassic', 'AIPlayer2', 'AIAgent3', 'AIAgent4',
    'NaiveAgent1', 'NaiveAgent2', 'NaiveAgent3', 'NaiveAgent4', 'NaiveAgent5',
    'NaiveAgent6', 'NaiveAgent7', 'NaiveAgent8', 'NaiveAgent9', 'NaiveAgent10',
    'UltimateBattleshipAgent'
]
