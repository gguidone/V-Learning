# ================================
# bandit.py
# ================================

import numpy as np

class HedgeBandit:
    """
    A simple Hedge (FTRL) adversarial-bandit subroutine.
    Maintains log-weights over actions; updates via unbiased loss estimates.
    """
    def __init__(self, num_actions, eta):
        self.num_actions = num_actions
        self.eta = eta
        # Log-space weights (initialized to zero)
        self.weights = np.zeros(num_actions)

    def get_distribution(self):
        # Numerically stable softmax: subtract max before exponentiating
        w = self.weights
        max_w = np.max(w)
        shifted = w - max_w               # now max(shifted)=0, so exp(shifted) ≤ 1
        exp_shifted = np.exp(shifted)
        total = np.sum(exp_shifted)
        if total == 0:
            # (Should rarely happen if you zero-initialized weights and apply small updates,
            # but just in case all weights become extremely negative, default to uniform.)
            return np.ones(self.num_actions) / self.num_actions
        return exp_shifted / total

    def sample_action(self):
        probs = self.get_distribution()
        return np.random.choice(self.num_actions, p=probs)

    def update(self, action_taken, loss):
        """
        Args:
            action_taken (int): Index of the action that was taken.
            loss (float): Observed scalar loss in [0, 1].
        Performs the standard Hedge update with unbiased loss estimate.
        """
        probs = self.get_distribution()
        loss_est = np.zeros(self.num_actions)
        loss_est[action_taken] = loss / probs[action_taken]
        self.weights -= self.eta * loss_est
