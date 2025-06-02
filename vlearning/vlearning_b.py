# ======================================
# vlearning.py  (memory-sparse + tqdm)
# ======================================

import numpy as np
from tqdm import trange, tqdm
from vlearning.ftrl_bandit import FTRLBandit
from collections import defaultdict
import bisect
from pettingzoo.classic import tictactoe_v3

class VLearning:
    """
    Memory-sparse V-Learning with a tqdm progress bar.
    Only stores visited (h, s, i) distributions and uses final visit counts
    to build the α-mixture, avoiding giant pre-allocations.
    """

    def __init__(self, env, num_episodes, H):
        self.env = env
        self.K = num_episodes
        self.agents = env.agents
        self.params = {}
        self.agent_action={}

        self.H = H
        # self.S = S
        # self.A = A

        for agent in self.agents:
            self.params[agent] = {
            'V_tilde': defaultdict(lambda:1), #init at 1s for tictactoe keys are (h,s)
            'V' : defaultdict(int), #keys are (h,s)
            'N_count' : defaultdict(list), #keys are (h,s), values are list of episodes when (h,s) visited. eg [1,5,6]. times visited = len of list

        # One HedgeBandit per (h, s)
            'bandits' : defaultdict(lambda: FTRLBandit(9,9,self.K)), #default dict with default value HedgeBandit(9,9). keys are (h,s). Maybe can reduce number of actions for increasing time step
            'policies': defaultdict(lambda: [np.array([1]*9)/9]) #initial policy is uniform over 9 actions. key is (h,s). length of list is number of times visited - 1. values are distribution
            }

    def alpha_t(self, t):
        return (self.H+1)/(self.H+t)

    def train(self):
        """
        Run K episodes of V-Learning with a tqdm progress bar.
        Whenever (h,s) is visited for the i-th time, we store the distribution
        in self.stored_distributions[(h, s, i)].
        """
        for k in trange(1, self.K + 1, desc="Training Episodes"):
            h = 0 #timestep
            self.env.reset(seed=42) #maybe remove seed to get randomness across episodes?
            observation, reward, termination, truncation, info = self.env.last() # rewards are for the next agent because after player 1 acts then it's player's 2 turn so the reward is for player 2's previous action.
            for agent in self.env.agent_iter():
                h+=1 #timestep starts at h = 1
                # print(f'h={h}')
                # observations are not the same for each agent. observation['observation']. Player 1 is +1, player 2 is -1. 0 is empty
                s =  self.env.observe(self.agents[0])['observation'][:,:,0] - self.env.observe(self.agents[0])['observation'][:,:,1]# unify observation using player_1 as reference.
                # print(s)
                key = (h,s.tobytes())
                if termination or truncation:
                    break
                else:
                    mask = observation["action_mask"]
                    # this is where you would insert your policy
                    # action = env.action_space(agent).sample(mask)
                    for player in self.agents:#sample actions from both players
                        action = self.params[player]['bandits'][key].sample_action()
                        self.agent_action[player] = action
                # print(f'action = {self.agent_action[agent]}')
                self.env.step(self.agent_action[agent])
                observation, reward, termination, truncation, info = self.env.last() # rewards are for the next agent because after player 1 acts then it's player's 2 turn so the reward is for player 2's previous action.
                next_s = self.env.observe(self.agents[0])['observation'][:,:,0] - self.env.observe(self.agents[0])['observation'][:,:,1]
                for player in self.agents: #update params for both agents
                    self.params[player]['N_count'][key].append(k) #append episode when (h,s) visited.
                    t = len(self.params[player]['N_count'][key]) #number of times (h,s) visited across episodes
                    alpha_t = (self.H+1)/(self.H+t)
                    beta_t = 0
                    next_V = self.params[player]['V'][(h+1,next_s.tobytes())] if h<self.H else 0 #0 Value if at terminal step
                    self.params[player]['V_tilde'][key] = (
                            (1-alpha_t)*self.params[player]['V_tilde'][key]) + alpha_t*(env.rewards[player]+next_V+beta_t)
                    self.params[player]['V'][key] = min(1, self.params[player]['V_tilde'][key])
                    self.params[player]['bandits'][key].update(self.agent_action[player],(1-env.rewards[player]-next_V)/1)
                    self.params[player]['policies'][key].append(self.params[player]['bandits'][key].get_distribution())#append distribution
            self.env.close()

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

    def sample_output_policy(self,state,h:int,k:int)->dict:
        # input k = np.random.randint(self.K) sampled from [0,K-1]. Use same k for all timestep
        # state should be from player_1 observation only for both players
        key = (h,state.tobytes())
        policies = {}
        for agent in self.agents:
            t = bisect.bisect_left(self.params[agent]['N_count'][key],k) #number of visits before kth episode
            alpha_probs = np.array([self.alpha_t(i) for i in range(0,t+1)]) #select i from [0,t]
            alpha_probs[0] = 1 #otherwise it's H+1/H which is wrong.
            prod = 1
            for idx in range(t, -1, -1):
                alpha_probs[idx] = alpha_probs[idx] * prod
                prod *= (1.0 - self.alpha_t(idx))
            # if alpha_probs.sum() != 1:
            #     print(alpha_probs.sum())
            # assert alpha_probs.sum() == 1# check that it is probability
            i = np.random.choice(t+1, p=alpha_probs/alpha_probs.sum()) #sample i from [0,t]. alpha_probs not exactly sum to 1 due to rounding errors.
            #episode when (h,s) visited for the ith time.
            # k = self.params[agent]['N_count'][key][i-1] #index from 0
            a = np.random.choice(9,p=self.params[agent]['policies'][key][i-1]) #policy before ith visit. hard coded action space size.
            policies[agent] = a
        return policies

env = tictactoe_v3.env()
env.reset(seed=42)
alg = VLearning(env,10,9)
alg.train()



