# ======================================
# vlearning.py  (memory-sparse + tqdm)
# ======================================

import numpy as np
from tqdm import trange, tqdm
from vlearning.bandit import HedgeBandit

class VLearning:
    """
    Memory-sparse V-Learning with a tqdm progress bar.  
    Only stores visited (h, s, i) distributions and uses final visit counts
    to build the α-mixture, avoiding giant pre-allocations.
    """

    def __init__(self, env, num_episodes):
        self.env = env
        self.K = num_episodes

        self.H = self.env.H
        self.S = self.env.get_num_states()
        self.A = self.env.get_num_actions()

        # Optimistic initialization of Ṽ and V
        self.V_tilde = np.zeros((self.H, self.S))
        self.V = np.zeros((self.H, self.S))
        self.N_count = np.zeros((self.H, self.S), dtype=int)

        for h in range(self.H):
            for s in range(self.S):
                self.V_tilde[h, s] = self.H + 1 - h
                self.V[h, s] = self.H + 1 - h

        # One HedgeBandit per (h, s)
        self.bandits = [
            [HedgeBandit(self.A, self.H) for _ in range(self.S)]
            for _ in range(self.H)
        ]

        # Only store distributions for visits that actually happen.
        # Key = (h, s, visit_index_i), value = length-A numpy array.
        self.stored_distributions = {}

    def train(self):
        """
        Run K episodes of V-Learning with a tqdm progress bar.
        Whenever (h,s) is visited for the i-th time, we store the distribution
        in self.stored_distributions[(h, s, i)].
        """
        for k in trange(1, self.K + 1, desc="Training Episodes"):
            s = self.env.reset()

            for h in range(self.H):
                dist = self.bandits[h][s].get_distribution()

                # Determine which visit-index this is at (h, s)
                visit_i = self.N_count[h, s] + 1

                # Store this distribution under key (h, s, i)
                self.stored_distributions[(h, s, visit_i)] = dist.copy()

                # Sample an action
                a = np.random.choice(self.A, p=dist)

                # Step in the environment
                s_next, r, done = self.env.step(a)

                # Update visit count
                t = visit_i
                self.N_count[h, s] = t

                # Learning rate α_t = (H + 1) / (H + t)
                alpha_t = (self.H + 1) / float(self.H + t)
                beta_t = 0.0  # no extra bonus

                # Next-step value
                V_next = 0.0 if (h + 1 == self.H) else self.V[h + 1, s_next]

                # Update Ṽ and V
                self.V_tilde[h, s] = (
                    (1 - alpha_t) * self.V_tilde[h, s]
                    + alpha_t * (r + V_next + beta_t)
                )
                self.V[h, s] = min(self.H + 1 - h, self.V_tilde[h, s])

                # One-step loss ℓ = (H - (r + V_next)) / H, clipped to [0,1]
                loss = (self.H - (r + V_next)) / float(self.H)
                loss = np.clip(loss, 0.0, 1.0)

                # Hedge update at (h, s)
                self.bandits[h][s].update(a, loss)

                s = s_next
                if done:
                    break

    def get_output_policy_sparse(self):
            """
            Build a sparse policy dict, showing a tqdm bar while iterating 
            over exactly the (h,s) pairs that were visited in training.
            Returns:
                sparse_policy: { h: { s: length-A numpy array } }
            """

            # 1) Extract unique (h,s) pairs from stored_distributions' keys
            visited_pairs = {(h, s) for (h, s, i) in self.stored_distributions.keys()}
            total = len(visited_pairs)
            print(f">>> Number of visited (h,s) pairs = {total}")
            # Initialize an empty nested dict only for visited entries
            sparse_policy = {h: {} for h in range(self.H)}

            # 2) Iterate over visited_pairs with a progress bar
            for (h, s) in tqdm(visited_pairs, desc="Building policy", total=total):
                t = self.N_count[h, s]  # number of visits

                # Build all α_i = (H+1)/(H + i) for i=1..t
                alpha_vec = np.array([(self.H + 1) / float(self.H + i) for i in range(1, t + 1)])

                # Now form the “product term” ∏_{j=i+1..t} (1 – α_j) in one backward pass:
                weights = np.empty(t, dtype=float)
                prod = 1.0
                for idx in range(t - 1, -1, -1):
                    # idx = i-1 in zero-based indexing, corresponding to “i = idx+1”
                    weights[idx] = alpha_vec[idx] * prod
                    prod *= (1.0 - alpha_vec[idx])

                # Normalize so that ∑_{i=1}^t weights[i-1] = 1
                weights /= weights.sum()

                # Mix all stored distributions for (h,s)
                dist_mix = np.zeros(self.A)
                for (i, w) in enumerate(weights, start=1):
                    dist_mix += w * self.stored_distributions[(h, s, i)]

                sparse_policy[h][s] = dist_mix

            return sparse_policy

