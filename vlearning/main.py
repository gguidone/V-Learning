# ================================
# main.py
# ================================

from vlearning.mdp import ChainMDP
from vlearning.vlearning import VLearning

if __name__ == "__main__":
    # Example 1: Chain MDP with 5 states, horizon 10
    chain_env = ChainMDP(N=5, H=10)
    vlearner = VLearning(env=chain_env, num_episodes=20000, c=1.0, eta=0.1, seed=42)
    vlearner.train()
    final_policy = vlearner.get_output_policy()
    print("Final policy at (h=0, s=0):", final_policy[0][0])
    final_greedy = vlearner.get_greedy_policy()
    print("Greedy Hedge policy at (h=0, s=0):", final_greedy[0][0])

    # You can add more test cases by creating other MDP subclasses and running similarly.
