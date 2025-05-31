# ================================
# vlearning.py
# ================================

import numpy as np
from bandit import HedgeBandit

class VLearning:
    """
    V-Learning implementation that accepts any BaseMDP environment.
    """
    def __init__(self, env, num_episodes, c=1.0, eta=0.1, seed=None):
        """
        Args:
            env (BaseMDP): Any MDP implementing the BaseMDP interface.
            num_episodes (int): Number of training episodes (K).
            c (float): Constant for the exploration bonus beta_t = c * sqrt(H^3 * ln(A) / t).
            eta (float): Initial learning-rate parameter for Hedge bandits.
            seed (int, optional): Random seed for reproducibility.
        """
        self.env = env
        self.K = num_episodes
        self.c = c
        self.eta = eta

        if seed is not None:
            np.random.seed(seed)

        # Extract dimensions from env
        self.H = self.env.H
        self.S = self.env.get_num_states()
        self.A = self.env.get_num_actions()

        # Initialize V_tilde, V, and visitation counts N_count[h][s]
        self.V_tilde = np.zeros((self.H, self.S))
        self.V = np.zeros((self.H, self.S))
        self.N_count = np.zeros((self.H, self.S), dtype=int)

        # Optimistic initialization: V_tilde[h][s] = H+1-h
        for h in range(self.H):
            for s in range(self.S):
                self.V_tilde[h, s] = self.H + 1 - h
                self.V[h, s] = self.H + 1 - h

        # Initialize Hedge bandits: one per (h, s)
        self.bandits = [
            [HedgeBandit(self.A, self.eta) for _ in range(self.S)]
            for _ in range(self.H)
        ]

        # To store the distribution (policy) at each visit for final output policy
        self.stored_distributions = {
            (h, s, i): None
            for h in range(self.H) for s in range(self.S) for i in range(self.K + 1)
        }
        # To store the visitation counts at the beginning of each episode
        self.N_count_by_episode = np.zeros((self.K + 1, self.H, self.S), dtype=int)

    def train(self):
        """
        Run K episodes of V-Learning on self.env.
        """
        for k in range(1, self.K + 1):
            # Record counts at beginning of episode k
            self.N_count_by_episode[k] = self.N_count.copy()

            # Reset environment
            s = self.env.reset()

            for h in range(self.H):
                # Current (h, s) bandit distribution
                dist = self.bandits[h][s].get_distribution()

                # Store distribution for this visit
                visit_number = self.N_count[h, s] + 1
                self.stored_distributions[(h, s, visit_number)] = dist.copy()

                # Sample action
                a = np.random.choice(self.A, p=dist)

                # Step in the environment
                s_next, r, done = self.env.step(a)

                # Update visit count
                t = self.N_count[h, s] + 1
                self.N_count[h, s] = t

                # Learning rate α_t = (H + 1)/(H + t)
                alpha_t = (self.H + 1) / float(self.H + t)
                # Bonus β_t = c * sqrt(H^3 * ln(A) / t)
                beta_t = self.c * np.sqrt((self.H ** 3) * np.log(self.A) / t)

                # V_{h+1}(s_next)
                V_next = 0.0 if (h + 1 == self.H) else self.V[h + 1, s_next]

                # Update V_tilde and V
                self.V_tilde[h, s] = (
                    (1 - alpha_t) * self.V_tilde[h, s]
                    + alpha_t * (r + V_next + beta_t)
                )
                self.V[h, s] = min(self.H + 1 - h, self.V_tilde[h, s])

                # One-step loss ℓ in [0,1]
                loss = (self.H - (r + V_next)) / float(self.H)
                loss = np.clip(loss, 0.0, 1.0)

                # Hedge update
                self.bandits[h][s].update(a, loss)

                # Move to next state
                s = s_next
                if done:
                    break

    def get_output_policy(self):
        """
        Constructs the final output policy \hat{\pi} after K episodes.
        Returns:
            policy (dict): A nested dict where policy[h][s] is a distribution over actions.
        """
        policy = {h: {s: np.zeros(self.A) for s in range(self.S)} for h in range(self.H)}

        for h in range(self.H):
            for s in range(self.S):
                t = self.N_count_by_episode[self.K, h, s]
                if t == 0:
                    # If (s, h) was never visited, default to uniform
                    policy[h][s] = np.ones(self.A) / self.A
                else:
                    # Compute mixture weights α_{i,t} for i ∈ {1..t}
                    alpha_list = []
                    for i in range(1, t + 1):
                        α_i = (self.H + 1) / float(self.H + i)
                        prod_term = 1.0
                        for j in range(i + 1, t + 1):
                            α_j = (self.H + 1) / float(self.H + j)
                            prod_term *= (1 - α_j)
                        alpha_list.append(α_i * prod_term)

                    alpha_array = np.array(alpha_list)
                    alpha_array /= alpha_array.sum()

                    # Mix the stored distributions
                    dist_mixture = np.zeros(self.A)
                    for i in range(1, t + 1):
                        dist_i = self.stored_distributions[(h, s, i)]
                        dist_mixture += alpha_array[i - 1] * dist_i

                    policy[h][s] = dist_mixture

        return policy
