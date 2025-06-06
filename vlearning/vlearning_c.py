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

    def __init__(self, env, num_episodes, H, beta_c,sym=False):
        self.env = env
        self.K = num_episodes
        self.agents = env.agents
        self.params = {}
        self.agent_action={}
        self.beta_c = beta_c
        self.sym=sym

        self.H = H
        # self.S = S
        # self.A = A

        for agent in self.agents:
            self.params[agent] = {
            'V_tilde': defaultdict(lambda:0.5*8+1), #keys are (h,s)
            'V' : defaultdict(lambda:0.5*8+1), #init 0.5 rewards for 8 step and win on final step
            'N_count' : defaultdict(list), #keys are (h,s), values are list of episodes when (h,s) visited. eg [1,5,6]. times visited = len of list

        # One bandit per (h, s)
            'bandits' : defaultdict(lambda: FTRLBandit(9,9,self.K)), #default dict with default value HedgeBandit(9,9). keys are (h,s). Maybe can reduce number of actions for increasing time step
            'policies': defaultdict(lambda: [np.array([1]*9)/9]) #initial policy is uniform over 9 actions. key is (h,s). length of list is number of times visited - 1. values are distribution
            }

    def alpha_t(self, t):
        return (self.H+1)/(self.H+t)

    # symmetries are all rotations plus reflection and rotation
    def get_sym(self,state,ret_set=True):
        if ret_set:
            symmetries = set({np.rot90(state, i).tobytes() for i in range(4)}).union(
                set({np.rot90(np.fliplr(state), i).tobytes() for i in range(4)}))
            return frozenset(symmetries)
        else:
            symmetries = []
            for i in range(4):
                if np.rot90(state, i).tobytes() not in symmetries:
                    symmetries.append(np.rot90(state, i).tobytes())
                if np.rot90(np.fliplr(state),i).tobytes() not in symmetries:
                    symmetries.append(np.rot90(np.fliplr(state),i).tobytes())
            return symmetries

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
            stop = False
            for agent in self.env.agent_iter():
                h+=1 #timestep starts at h = 1
                if h == 10:#terminate because game ended
                    stop = True
                # observations are not the same for each agent. observation['observation']. Player 1 is +1, player 2 is -1. 0 is empty
                s =  self.env.observe(self.agents[0])['observation'][:,:,0] - self.env.observe(self.agents[0])['observation'][:,:,1]# unify observation using player_1 as reference.
                key = (h,self.get_sym(s)) if self.sym else (h,s.tobytes())
                # mask = observation["action_mask"]
                # this is where you would insert your policy
                # action = env.action_space(agent).sample(mask)
                if stop:#update params for current state = terminal state then end game
                    """
                    Not sure if update params for current state = terminal state needed.
                    """
                    # for player in self.agents:  # update params for both agents
                    #     self.params[player]['N_count'][key].append(k)  # append episode when (h,s) visited.
                    #     t = len(self.params[player]['N_count'][key])  # number of times (h,s) visited across episodes
                    #     alpha_t = (self.H + 1) / (self.H + t)
                    #     beta_t = self.beta_c * np.sqrt(
                    #         9 ** 3 * 9 * np.log(9 * (3 ** 9) * 9 * self.K / 0.01) / t)  # delta = 0.01
                    #     reward = 0
                    #     self.params[player]['V_tilde'][key] = ((1 - alpha_t) * self.params[player]['V_tilde'][key]) + alpha_t * (reward + next_V + beta_t)
                    #     self.params[player]['V'][key] = min(0.5*8+1, self.params[player]['V_tilde'][key])
                    #     # print("Reward for player", player, ":", env.rewards[player])
                    #     self.params[player]['bandits'][key].update(self.agent_action[player], (0.5*8+1-reward-next_V)/(0.5*8+1))
                    #     self.params[player]['policies'][key].append(
                    #         self.params[player]['bandits'][key].get_distribution())  # append distribution
                    break
                else:
                    for player in self.agents:#sample actions from both players
                        if self.sym:
                            action = None
                            sym_list = self.get_sym(s,False)
                            # sample bandit with equivalent states. Need to transform mask to match state transformations
                            for i in sym_list:
                                # print('sample')
                                # print(
                                #     f'state = {np.frombuffer(i, dtype=np.int8)}, mask = {np.frombuffer(mask, dtype=np.int8)}')
                                if action is None:#set action to be sampled from bandit using original state with not transformations
                                    action = self.params[player]['bandits'][(h,i)].sample_action(
                                    mask=1-abs(np.frombuffer(i,dtype=np.int8)))
                                else:
                                    self.params[player]['bandits'][(h, i)].sample_action(
                                        mask=1-abs(np.frombuffer(i,dtype=np.int8)))
                        else:
                            action =  self.params[player]['bandits'][key].sample_action(mask = self.env.observe(player)['action_mask']) #sample action from bandit, optional mask
                        self.agent_action[player] = action
                    self.env.step(self.agent_action[agent])
                observation, reward, termination, truncation, info = self.env.last() # rewards are for the next agent because after player 1 acts then it's player's 2 turn so the reward is for player 2's previous action.
                next_s = self.env.observe(self.agents[0])['observation'][:,:,0] - self.env.observe(self.agents[0])['observation'][:,:,1]
                for player in self.agents: #update params for both agents
                    self.params[player]['N_count'][key].append(k) #append episode when (h,s) visited.
                    t = len(self.params[player]['N_count'][key]) #number of times (h,s) visited across episodes
                    alpha_t = (self.H+1)/(self.H+t)
                    beta_t = self.beta_c*np.sqrt(9**3*9*np.log(9*(3**9)*9*self.K/0.01)/t) #delta = 0.01
                    if np.array_equal(s,next_s):# s == next_s when illegal action selected
                        next_V=0
                        reward = self.env.rewards[player] + 1#illegal action by agent gets 0, other player gets 1
                        stop = True
                    else:
                        next_V = self.params[player]['V'][(h+1,self.get_sym(next_s))] if self.sym else self.params[player]['V'][(h+1,next_s.tobytes())]
                        if self.env.rewards['player_1'] == self.env.rewards['player_2']:#game continues
                            reward = 0.5
                        else:
                            reward = (self.env.rewards[player] + 1)/2 #game ends without illegal move
                            stop = True
                    self.params[player]['V_tilde'][key] = (
                            (1-alpha_t)*self.params[player]['V_tilde'][key]) + alpha_t*(reward+next_V+beta_t)
                    self.params[player]['V'][key] = min(0.5*8+1, self.params[player]['V_tilde'][key])
                    if self.sym:#update all bandits which are symmetric to each other
                        sym_list = self.get_sym(s, False)
                        for i in sym_list:
                            # print('update')
                            # print(f'state = {np.frombuffer(i,dtype=np.int8)}, mask = {np.frombuffer(mask,dtype=np.int8)}')
                            self.params[player]['bandits'][(h,i)].update(self.agent_action[player],
                                                                   (0.5 * 8 + 1 - reward - next_V) / (0.5 * 8 + 1))
                            self.params[player]['policies'][(h,i)].append(#store policies on a per state basis, not group of symmetry states
                                self.params[player]['bandits'][(h,i)].get_distribution(mask=1-abs(np.frombuffer(i,dtype=np.int8))))
                    else:
                        self.params[player]['bandits'][key].update(self.agent_action[player],(0.5*8+1-reward-next_V)/(0.5*8+1))
                        self.params[player]['policies'][key].append(self.params[player]['bandits'][key].get_distribution(mask=1-abs(s.flatten())))#append distribution. mask from current state, not next state
            self.env.close()

    def sample_output_policy(self,state,h:int,k:int)->dict:
        # input k = np.random.randint(self.K) sampled from [0,K-1]. Use same k for all timestep
        # state should be from player_1 observation only for both players
        key = (h,self.get_sym(state)) if self.sym else (h,state.tobytes())
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
            a = np.random.choice(9,p=self.params[agent]['policies'][(h,state.tobytes())][i-1]) if self.sym else np.random.choice(9,p=self.params[agent]['policies'][key][i-1]) #policy before ith visit. hard coded action space size.
            policies[agent] = a
        return policies

def get_state(env):
    return env.observe('player_1')['observation'][:, :, 0] - env.observe('player_1')['observation'][:, :, 1]

def play(vlearn=None,opponent = 'player_2',watch = False,eval=False,eps=1e6):
    env = tictactoe_v3.env()
    env.reset(seed=42)
    assert opponent in env.agents
    h=0
    if vlearn is not None:
        k = {'player_1': np.random.randint(alg.K),
             'player_2': np.random.randint(alg.K)}  # for output policy, different random seed for product policy
        if eval:
            res = defaultdict(int)
            for ep in trange(1, eps + 1, desc="Simulating Episodes"):
                env.reset()
                h=0
                k = {'player_1': np.random.randint(alg.K),
                     'player_2': np.random.randint(alg.K)}
                for agent in env.agent_iter():
                    h+=1
                    observation, reward, termination, truncation, info = env.last()
                    s = env.observe('player_1')['observation'][:, :, 0] - env.observe('player_1')['observation'][:, :,1]
                    if termination:
                        if env.rewards['player_1'] == env.rewards['player_2']:
                            res['draw']+=1
                        elif env.rewards['player_1'] > env.rewards['player_2']:
                            res['player 1 wins']+=1
                        else:
                            res['player 2 wins']+=1
                        break

                    action = alg.sample_output_policy(s, h, k[agent])[agent]
                    env.step(action)
            return res

        if watch:
            for agent in env.agent_iter():
                h+=1
                observation, reward, termination, truncation, info = env.last()
                s =  env.observe('player_1')['observation'][:, :, 0] - env.observe('player_1')['observation'][:, :, 1]
                print(s)
                if termination:
                    if env.rewards['player_1'] == env.rewards['player_2']:
                        print('draw')
                    elif env.rewards['player_1'] > env.rewards['player_2']:
                        print('player 1 wins')
                    else:
                        print('player 2 wins')
                    break

                action = alg.sample_output_policy(s, h, k[agent])[agent]
                print(f'{agent} chooses action {action}')
                env.step(action)

        else:
            for agent in env.agent_iter():
                h+=1
                observation, reward, termination, truncation, info = env.last()
                s =  env.observe('player_1')['observation'][:, :, 0] - env.observe('player_1')['observation'][:, :, 1]
                print(s)
                if termination:
                    if env.rewards['player_1'] == env.rewards['player_2']:
                        print('draw')
                    elif env.rewards[opponent] < 0:
                        print('you win')
                    else:
                        print('you lose')
                    break
                if opponent != agent:
                    print('your move')
                    action = int(input())
                    env.step(action)
                else:
                    action = alg.sample_output_policy(s,h,k[opponent])[opponent]
                    print(f'opponent chooses action {action}')
                    env.step(action)
        env.close()

    else:
        res = defaultdict(int)
        for ep in range(int(eps)):
            env.reset()

            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                if termination or truncation:
                    if env.rewards['player_1'] == env.rewards['player_2']:
                        res['draw']+=1
                    elif env.rewards['player_1'] == 1:
                        res['player 1 wins']+=1
                    else:
                        res['player 2 wins']+=1
                    break

                else:
                    mask = observation["action_mask"]
                    # this is where you would insert your policy
                    action = env.action_space(agent).sample(mask)
                env.step(action)
            env.close()
        return res
en = tictactoe_v3.env()
en.reset(seed=42)
# K=10460353203*5
alg = VLearning(en,10**6,9,0.01,True)
alg.train()
res = play(alg,eval=True,eps = 1000)

#epsilon bound. delta = 0.01
# print(np.sqrt(9**5*(3**9)*9*np.log(9*(3**9)*9*K/0.01)/K))
#baseline {'player 1 wins': 585692, 'player 2 wins': 287383, 'draw': 126925}