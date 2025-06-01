# =====================================
# train_tictactoe.py
# =====================================

import pickle
from tictactoe.run_tictactoe import run_two_player_tictactoe

def train_and_save(K=20000, seed=2025):
    """
    Train X and O for K episodes, then save their mixed‐policy dicts to disk.
    
    - policy_X.pkl will contain X’s final policy dict.
    - policy_O.pkl will contain O’s final policy dict.
    """
    # Run V‐Learning for both players (in lockstep)
    policy_X, policy_O = run_two_player_tictactoe(K=K, seed=seed)
    
    # Save X’s policy
    with open('policy_X.pkl', 'wb') as f:
        pickle.dump(policy_X, f)
    print("Saved policy_X.pkl")
    
    # Save O’s policy
    with open('policy_O.pkl', 'wb') as f:
        pickle.dump(policy_O, f)
    print("Saved policy_O.pkl")

if __name__ == "__main__":
    # You can change K or seed as you like
    train_and_save(K=200000, seed=2025)
