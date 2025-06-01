# =====================================
# tictactoe_mdp.py
# =====================================

from abc import ABC, abstractmethod
from vlearning.mdp import BaseMDP

class TicTacToeMDP(BaseMDP):
    """
    A two‐player zero‐sum Tic-Tac-Toe MDP.  We show the implementation as "one agent at a time"—
    you must run two copies (player_id=1 for X, player_id=2 for O) in lockstep, exactly like Matching Pennies.

    State encoding:
      - We keep a 3×3 board internally as a list of 9 integers in {0,1,2}.
         0 = empty, 1 = X, 2 = O.
      - We also store whose turn it is: turn = 1 for X, turn = 2 for O.
      - We flatten (board, turn) to a single integer (0 ≤ s < 3^9 * 2) for tabular indexing.
        (That is, there are 2 * 3^9 possible “states,” though many are unreachable; for simplicity
         we still allocate arrays of that full size.)
      - `action` is an integer from 0..8, indicating which cell (row=action//3, col=action%3) to place
        this player’s piece.  Illegal moves (action on a non‐empty cell) yield a big negative reward
        and end the episode immediately.

    Rewards:
      - If this player places and **wins** (completes three‐in‐a‐row), reward = +1, done = True.
      - Else if the board is full (9 moves played) ⇒ draw ⇒ reward = 0, done = True.
      - Else if this player’s move is illegal (cell not empty) ⇒ reward = –1, done = True.
      - Otherwise: reward = 0, done = False, and turn passes to the other player.

    Horizon H = 9 (maximum 9 moves total).
    """

    def __init__(self, player_id):
        """
        Args:
          player_id (int): 1 for X (first mover) or 2 for O (second mover).
        """
        assert player_id in (1, 2)
        self.player_id = player_id
        self.other_id = 2 if player_id == 1 else 1

        # We store the board as a list of length 9, each ∈ {0,1,2}.
        # 0 = empty; 1 = X; 2 = O.
        self.board = [0] * 9

        # Whose turn is it? 1 = X’s turn; 2 = O’s turn.
        # At reset(), we always start with X’s turn.
        self.turn = 1

        # Move count: how many moves have been played so far (0..9).
        self.move_count = 0

        # The horizon is at most 9 moves.
        self._H = 9

    @property
    def H(self):
        return self._H

    def reset(self):
        """
        Clear the board, set turn = 1 (X), move_count = 0.
        Return the initial state encoding.
        """
        self.board = [0] * 9
        self.turn = 1
        self.move_count = 0
        return self._encode_state()

    def step(self, action):
        """
        Apply this player’s action (0..8) if it is their turn.  Because we run two copies
        in lockstep, we only allow this agent to move if self.turn == self.player_id; otherwise
        that call should never happen in a correct driver.
        """
        # If it’s not this player’s turn, that is a usage error.
        if self.turn != self.player_id:
            raise RuntimeError("TicTacToeMDP.step(...) called by wrong player")

        # Check legality:
        if action < 0 or action >= 9 or self.board[action] != 0:
            # Illegal move ⇒ immediate loss for this agent.
            # We set reward = –1, done = True.  Turn does not matter now.
            return self._encode_state(), -1.0, True

        # Place this player's mark on the board:
        self.board[action] = self.player_id
        self.move_count += 1

        # Check if this move wins:
        if self._is_winner(self.player_id):
            # This player wins: +1
            return self._encode_state(), 1.0, True

        # If board is full ⇒ draw:
        if self.move_count == 9:
            return self._encode_state(), 0.0, True

        # Otherwise: no immediate reward (0), pass turn to the other player and continue.
        self.turn = self.other_id
        return self._encode_state(), 0.0, False

    def get_num_states(self):
        # We encode (board, turn) as an integer in [0, 3^9 * 2).
        return 2 * (3 ** 9)

    def get_num_actions(self):
        # Always 9 possible “places,” even if illegal.
        return 9

    # ---------- Internals for encoding/decoding states ----------

    def _encode_state(self):
        """
        Convert (self.board, self.turn) → an integer 0 <= s < 2*3^9.
        We treat the board as a base-3 number, then add either 0 or 3^9 depending on turn.
        """
        code = 0
        for cell in self.board:
            code = code * 3 + cell
        # If turn == 2 (O), add 3^9 offset
        if self.turn == 2:
            code += 3 ** 9
        return code

    def _decode_state(self, code):
        """
        Convert integer back to (board, turn).  Not strictly needed unless you want to
        inspect states. We do the inverse of _encode_state().
        """
        turn_bit = 0
        if code >= 3 ** 9:
            turn_bit = 1
            code -= 3 ** 9
        board = [0] * 9
        for i in reversed(range(9)):
            board[i] = code % 3
            code //= 3
        turn = 2 if turn_bit == 1 else 1
        return board, turn

    def _is_winner(self, pid):
        """
        Check if player 'pid' (1 or 2) has three in a row on the current board.
        """
        b = self.board
        wins = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]
        for (i, j, k) in wins:
            if b[i] == pid and b[j] == pid and b[k] == pid:
                return True
        return False
