# ======================================
# vlearning.py  (memory-sparse + tqdm)
# ======================================

import numpy as np
from tqdm import trange, tqdm
from vlearning.ftrl_bandit import FTRLBandit
#from vlearning.bandit import HedgeBandit

class VLearning:
    """
    Memory-sparse V-Learning with a tqdm progress bar.  
    Only stores visited (h, s, i) distributions and uses final visit counts
    to build the α-mixture, avoiding giant pre-allocations.
    """
    def __init__(self, env, num_episodes):
        self.env = env
        self.K = num_episodes

        # H = horizon (number of time-steps per episode)
        # S = number of distinct states in this MDP
        # A = number of possible actions in each state
        self.H = self.env.H
        self.S = self.env.get_num_states()
        self.A = self.env.get_num_actions()

        # Ṽ[h,s] = “auxiliary” value estimate at step h, state s
        # V[h,s]  = clipped value estimate (≤ H+1−h)
        # N_count[h,s] = how many times (h,s) has been visited so far
        self.V_tilde = np.zeros((self.H, self.S))
        self.V       = np.zeros((self.H, self.S))
        self.N_count = np.zeros((self.H, self.S), dtype=int)

        # Optimistic initialization: at each (h,s), set Ṽ = V = (H+1−h).
        # That way, before any updates, Ṽ[h,s] = “max possible future return.”
        for h in range(self.H):
            for s in range(self.S):
                self.V_tilde[h, s] = 0
                self.V[h, s]       = 1 #self.H + 1 - h

        # Create a HedgeBandit (EXP3-style) for each (h, s).
        # bandits[h][s] will learn a mixture over A actions at that (h,s).
        # Instead, initialize an empty dict for each h:
        self.bandits = [dict() for _ in range(self.H)]

        # stored_distributions[(h, s, i)] = the i-th mixed distribution π_{h,s}^{(i)}.
        # We only insert a key here when (h,s) is actually visited the i-th time.
        self.stored_distributions = {}

    def train(self):
        """
        Run K episodes of V-Learning with a tqdm progress bar.
        Whenever (h,s) is visited for the i-th time, we store the distribution
        in self.stored_distributions[(h, s, i)].
        """
        iota = np.log(self.H * self.S * self.A * self.K/0.9)  # for the β_t term)
        # trange(1, K+1) shows a live “Training Episodes: XX% …” bar.
        for k in trange(1, self.K + 1, desc="Training Episodes"):
            # Reset environment; get the initial state code s (an integer in [0..S−1])
            s = self.env.reset()

            # Run one full episode of length ≤ H time-steps
            for h in range(self.H):
                # --- Ensure the bandit for (h,s) exists ---
                if s not in self.bandits[h]:
                    self.bandits[h][s] = FTRLBandit(self.A, self.H, self.K)

                # 1) Ask the bandit for its current θₜ:
                dist = self.bandits[h][s].get_distribution()

                # 2) Store that θₜ for mixture‐extraction later:
                visit_i = self.N_count[h, s] + 1
                self.stored_distributions[(h, s, visit_i)] = dist.copy()

                # 3) <<<<<<<<<< NEW >>>>>>>>>>
                #    Sample directly from the bandit. This does:
                #       (a) compute θₜ via softmax(C_{t−1})
                #       (b) draw aₜ ∼ θₜ
                #       (c) increment bandit.current_round from t−1 → t
                a = self.bandits[h][s].sample_action()

                # 4) Step the environment with action a:
                s_next, r, done = self.env.step(a)

                # 5) Update visit‐count for (h,s):
                t = visit_i
                self.N_count[h, s] = t

                # 6) Do your V‐learning TD‐type update exactly as before:
                alpha_t = (self.H + 1) / float(self.H + t)
                V_next  = 0.0 if (h + 1 == self.H) else self.V[h + 1, s_next]
                self.V_tilde[h, s] = (
                    (1 - alpha_t) * self.V_tilde[h, s]
                    + alpha_t * (r + V_next)
                )
                self.V[h, s] = min(self.H + 1 - h, self.V_tilde[h, s])

                # 7) Compute one‐step loss ∈ [0,1]:
                loss = (self.H - (r + V_next)) / float(self.H)
                loss = np.clip(loss, 0.0, 1.0)

                # 8) <<<<<<<<<< NEW >>>>>>>>>>
                #    Now that bandit.current_round == t, update it with the observed loss:
                self.bandits[h][s].update(a, loss)

                # 9) Prepare for next step (or break if done)
                s = s_next
                if done:
                    break


    def get_output_policy(self):
        """
        Build a sparse policy dict, showing a tqdm bar while iterating 
        over exactly the (h,s) pairs that were visited in training.
        Returns:
            sparse_policy: { h: { s: length-A numpy array } }
        """

        # 1) Extract exactly the (h,s) pairs that were visited at least once:
        visited_pairs = { (h, s) for (h, s, i) in self.stored_distributions.keys() }
        total = len(visited_pairs)
        print(f">>> Number of visited (h,s) pairs = {total}")

        # Prepare an empty dict-of-dicts: we will only fill in the visited (h,s)
        sparse_policy = { h: {} for h in range(self.H) }

        # 2) Loop over each visited (h,s) with a tqdm bar
        for (h, s) in tqdm(visited_pairs, desc="Building policy", total=total):
            t = self.N_count[h, s]   # how many times (h,s) was seen

            # (a) Build the “bare” alpha vector: α_i = (H+1)/(H + i), for i=1..t
            alpha_vec = np.array([ (self.H + 1) / float(self.H + i) 
                                   for i in range(1, t+1) 
                                 ])

            # (b) Build w_i = α_i * ∏_{j=i+1..t} (1 − α_j) in a single backward pass:
            weights = np.empty(t, dtype=float)
            prod = 1.0
            # idx = i−1 in zero‐based indexing, so idx runs from (t−1) down to 0
            for idx in range(t - 1, -1, -1):
                weights[idx] = alpha_vec[idx] * prod
                prod *= (1.0 - alpha_vec[idx])

            # (c) Normalize so the weights sum to 1.0
            weights /= weights.sum()

            # (d) Now form the final mixed distribution at (h,s):
            dist_mix = np.zeros(self.A)
            # sum over i = 1..t of w_i * stored_distributions[(h,s,i)]
            for (i, w) in enumerate(weights, start=1):
                dist_mix += w * self.stored_distributions[(h, s, i)]

            # (e) Store that 9-vector under sparse_policy[h][s].
            sparse_policy[h][s] = dist_mix

        return sparse_policy


