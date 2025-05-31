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
        exp_w = np.exp(self.weights)
        return exp_w / np.sum(exp_w)

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
