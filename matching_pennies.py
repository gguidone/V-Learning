# ================================
# matching_pennies.py
# ================================
import numpy as np

class MatchingPennies:
    """
    A trivial two‐player zero‐sum game “MDP” of horizon=1:
      - Action space for each player is {0=Heads, 1=Tails}.
      - If both actions match, player1 gets reward +1, player2 gets –1.
      - If they differ, player1 gets –1, player2 gets +1.
      - There is no state change (always the same “round”).
    """
    def __init__(self):
        # We only have one “round,” so no persistent state is needed.
        pass

    def step(self, action1, action2):
        """
        Both players play simultaneously. Return (r1, r2).
        Args:
          action1 (int): Player 1’s choice (0 or 1).
          action2 (int): Player 2’s choice (0 or 1).
        Returns:
          (r1, r2): tuple of rewards for (player1, player2), each in {+1, –1}.
        """
        if action1 == action2:
            return +1, -1
        else:
            return -1, +1
