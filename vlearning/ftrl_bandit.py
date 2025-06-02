# =================================
# vlearning/ftrl_bandit.py
# =================================
import numpy as np

class FTRLBandit:
    """
    Incremental FTRL bandit (Algorithm 5 + Corollary 19) without
    precomputing full T‐length arrays.  Computes α_t, w_t, η_t on the fly.

    - B = num_actions
    - H = MDP horizon
    - T_max is only used as an upper bound (we assert we never exceed it).
    """
    def __init__(self, num_actions, horizon_H, T_max):
        self.B = num_actions
        self.H = horizon_H
        self.T = T_max

        # C_t[b] = cumulative, weighted loss ∑_{i=1..t} w_i·tildeℓ_i(b).
        self.C = np.zeros(self.B)

        # last_distribution = θₜ (for the current round).  Initialize to uniform.
        self.last_distribution = np.ones(self.B) / float(self.B)

        # How many updates/rounds we have done so far (= t).
        self.current_round = 0

        # We will maintain:
        #   prod_1_minus = ∏_{i=2..t} (1 − α_i)  (for the current t),
        #   so that w_t = α_t / prod_1_minus[t].
        # At t=1, define prod_1_minus := 1 (empty product).
        self.prod_1_minus = 1.0

    def get_distribution(self):
        """
        Return θ_{t}(·) where t = self.current_round + 1.
        (If current_round=0, this returns uniform θ₁.)
        """
        t_next = self.current_round + 1
        if t_next == 1:
            # Round 1: uniform
            return self.last_distribution.copy()

        # Otherwise, we just finished round k = current_round,
        # and updated C[b] = ∑_{i=1..k} w_i·tildeℓ_i(b).
        # We want θ_{k+1}(b) ∝ exp( − (η_k / w_k) · C_t[b] ).
        k = self.current_round
        # Compute α_k and w_k on the fly (we saved prod_1_minus for t=k).
        alpha_k = float(self.H + 1) / float(self.H + k)
        # For k=1, prod_1_minus was 1; for k>1, it is ∏_{i=2..k} (1 − α_i).
        w_k = alpha_k / self.prod_1_minus

        # Compute η_k = sqrt((H·ln B)/(B·k))
        eta_k = np.sqrt((self.H * np.log(self.B)) / float(self.B * k))

        factor = eta_k / w_k
        scores = - factor * self.C  # length‐B

        # softmax( scores )
        m = np.max(scores)
        exp_shifted = np.exp(scores - m)
        total = np.sum(exp_shifted)
        if total == 0.0:
            dist = np.ones(self.B) / float(self.B)
        else:
            dist = exp_shifted / total

        self.last_distribution = dist.copy()
        return dist

    def sample_action(self):
        """
        Sample aₜ ∼ θₜ(·), then increment current_round → t.
        """
        pi_t = self.get_distribution()
        a_t = np.random.choice(self.B, p=pi_t)
        self.current_round += 1

        # Now we need to update prod_1_minus to become ∏_{i=2..t} (1 − α_i).
        # Compute α_t = (H+1)/(H + t), then multiply into prod_1_minus.
        t = self.current_round
        assert 1 <= t <= self.T, "round t out of [1..T_max]"
        alpha_t = float(self.H + 1) / float(self.H + t)
        if t == 1:
            # by definition, prod_1_minus stays 1 after t=1,
            # since ∏_{i=2..1} is empty.  Then for t=2, we do prod_1_minus *= (1−α₂).
            self.prod_1_minus = 1.0
        else:
            # Multiply by (1 − α_t) to incorporate this round
            self.prod_1_minus *= (1.0 - alpha_t)

        return a_t

    def update(self, action_taken, loss):
        """
        Given that we just sampled a_t=action_taken on round t=self.current_round,
        form unbiased tildeℓ_t and update C[b] += w_t · tildeℓ_t(b).

        Args:
          action_taken (int): the chosen action index
          loss (float): observed loss ∈ [0,1]

        After sample_action(), self.current_round = t, so 1 ≤ t ≤ T_max.
        """
        t = self.current_round
        assert 1 <= t <= self.T, "update() must be called once per round in [1..T_max]"

        # θ_t(·) was already stored in last_distribution by get_distribution().
        pi_t = self.last_distribution
        p_at = max(pi_t[action_taken], 1e-12)

        # γ_t = η_t = sqrt((H·ln B)/(B·t))
        eta_t = np.sqrt((self.H * np.log(self.B)) / float(self.B * t))
        gamma_t = eta_t

        # Form tildeℓ_t(b) = 0 except at b=action_taken:
        # tildeℓ_t(action) = loss / (p_at + γ_t)
        denom = p_at + gamma_t
        tilde_loss = np.zeros(self.B)
        tilde_loss[action_taken] = loss / denom

        # w_t = α_t / (∏_{i=2..t}(1 − α_i)) = α_t / prod_1_minus[t]
        # But after sample_action(), we already did prod_1_minus *= (1 − α_t).
        # So the value of prod_1_minus right now is ∏_{i=2..t}(1 − α_i).
        alpha_t = float(self.H + 1) / float(self.H + t)
        w_t = alpha_t / self.prod_1_minus

        # Finally update C[b] += w_t · tildeℓ_t(b)
        self.C += w_t * tilde_loss

        # Done; next get_distribution() will use this updated C.
        return
