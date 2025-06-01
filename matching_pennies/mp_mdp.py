# =================================
# mp_mdp.py
# =================================

from abc import ABC, abstractmethod
from vlearning.mdp import BaseMDP

class MatchingPenniesMDP(BaseMDP):
    """
    A one-step (H=1) zero-sum matching pennies game, specialized for *one* agent at a time.
    This class will be used twice (once for P1, once for P2). Internally, it keeps track
    of both players' actions so that it can return the correct reward for THIS agent.
    """

    def __init__(self, player_id):
        """
        Args:
          player_id (int): 1 or 2, indicating which player this MDP instance is for.
        """
        assert player_id in (1, 2)
        self.player_id = player_id
        self.other_id = 2 if player_id == 1 else 1

        # We only have 1 state: index 0
        # And horizon H = 1.
        self._state = 0
        self._H = 1

        # We will store the two actions in this episode for both players.
        self._a1 = None
        self._a2 = None

    def reset(self):
        """
        Reset for a new episode. Always returns state=0.
        """
        self._state = 0
        return self._state

    def step(self, action):
        """
        One step: this agent picks `action` ∈ {0,1}. We assume the other agent's action
        was already chosen earlier in the same "time" by the other VLearner. To coordinate
        that, we'll use a short global handshake: the first call to step(...) in this episode
        sets this player's action; the second call sets the other player's action and resolves
        both rewards.
        
        For simplicity, we require that *both* VLearning instances call step() in the same
        order (first P1.step(a1), then P2.step(a2) OR vice versa), so we know who is first.
        """

        if self.player_id == 1:
            # Player 1 calls step(a1) first.
            self._a1 = action
            # Return dummy (next_state=0, reward=0, done=False). We only finalize after P2 steps.
            return 0, 0.0, False

        else:
            # Player 2 calls step(a2) second.
            self._a2 = action
            # Now that both a1 and a2 are known, compute rewards:
            if self._a1 == self._a2:
                r1, r2 = +1.0, -1.0
            else:
                r1, r2 = -1.0, +1.0
            # Return THIS agent's reward:
            reward = r2 if self.player_id == 2 else r1
            done = True
            # Next state is irrelevant (always 0), and episode ends (done=True).
            return 0, reward, done

    def get_num_states(self):
        return 1  # only state index 0

    def get_num_actions(self):
        return 2  # {Heads=0, Tails=1}

    @property
    def H(self):
        return self._H
