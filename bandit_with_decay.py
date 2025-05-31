# ================================
# bandit_with_decay.py
# ================================
import numpy as np

class HedgeBandit:
    """
    Hedge (EXP3) bandit that uses the paper's recommended time-decaying η_t:
        η_t = sqrt( (H * ln(B)) / (B * t) ).
    Each update uses the current visit count to compute η_t.
    """
    def __init__(self, num_actions, horizon_H):
        """
        Args:
          num_actions (int):  number of actions (B = |A|).
          horizon_H   (int):  the MDP horizon H (same for every (h,s)).
        """
        self.B = num_actions
        self.H = horizon_H

        # Log-weights w[i] for i in {0,1,...,B-1}
        self.weights = np.zeros(self.B)

        # How many times we've called update() so far (visit count t)
        self.visit_count = 0

    def get_distribution(self):
        """
        Numerically stable softmax over log-weights.
        Returns a length-B probability vector π.
        """
        w = self.weights
        m = np.max(w)             # subtract max to prevent overflow
        shifted = w - m           
        exp_shifted = np.exp(shifted)
        total = np.sum(exp_shifted)
        if total == 0:
            # If everything underflowed to 0, return uniform to avoid NaNs
            return np.ones(self.B) / self.B
        return exp_shifted / total

    def sample_action(self):
        """
        Sample a single action index in {0,...,B-1} according to the current π.
        """
        probs = self.get_distribution()
        return np.random.choice(self.B, p=probs)

    def get_eta(self):
        """
        Compute the time-decaying learning rate η_t = sqrt( (H * ln B)/(B * t) ).
        Must be called *after* incrementing self.visit_count.
        """
        t = self.visit_count
        # To avoid division by zero, ensure t >= 1
        assert t >= 1, "visit_count must be incremented before computing η_t"
        return np.sqrt((self.H * np.log(self.B)) / (self.B * t))

    def update(self, action_taken, loss):
        """
        Perform one Hedge update with time-decaying η_t.

        Args:
          action_taken (int): index of the action that was chosen.
          loss (float): observed loss in [0,1] for that action.
        """
        # 1) Increment visit count first (so t starts at 1)
        self.visit_count += 1
        t = self.visit_count

        # 2) Compute the current Hedge distribution π_t
        probs = self.get_distribution()
        p_a = max(probs[action_taken], 1e-12)  # avoid division by zero

        # 3) Form the unbiased loss-estimate vector ℓ̂_t
        loss_est = np.zeros(self.B)
        loss_est[action_taken] = loss / p_a

        # 4) Compute η_t according to the paper’s formula
        eta_t = np.sqrt((self.H * np.log(self.B)) / (self.B * t))

        # 5) Update log-weights
        self.weights -= eta_t * loss_est
