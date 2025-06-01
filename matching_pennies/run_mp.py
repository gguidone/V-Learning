# =================================
# run_mp.py
# =================================

import numpy as np
from mp_mdp import MatchingPenniesMDP
from vlearning.vlearning import VLearning

def run_two_player_vlearning(K=5000, seed=None):
    """
    Run two V-Learning agents on Matching Pennies (H=1).
    Each agent’s environment returns that agent’s reward.
    Returns the final mixed policy (a length-2 vector) for each player.
    """
    if seed is not None:
        np.random.seed(seed)

    # Create two MDPs, one for each player
    env1 = MatchingPenniesMDP(player_id=1)
    env2 = MatchingPenniesMDP(player_id=2)

    # Create two VLearning instances (5000 episodes each)
    learner1 = VLearning(env=env1, num_episodes=K)
    learner2 = VLearning(env=env2, num_episodes=K)

    # We run episodes “in lockstep,” manually updating both learners
    for episode in range(1, K + 1):
        # ------------ PLAYER 1’s “episode” ------------
        s1 = env1.reset()  
        # h = 0 (only one step in Matching Pennies)
        dist1 = learner1.bandits[0][s1].get_distribution()
        a1 = np.random.choice(learner1.A, p=dist1)

        # Take X’s “step” (dummy until P2 also steps)
        s1_next, r1_dummy, done1 = env1.step(a1)
        # (We’ll finalize r1 only after P2 steps)

        # ------------ PLAYER 2’s “episode” ------------
        s2 = env2.reset()
        dist2 = learner2.bandits[0][s2].get_distribution()
        a2 = np.random.choice(learner2.A, p=dist2)

        # Now P2 steps, which resolves both rewards
        s2_next, r2, done2 = env2.step(a2)
        _, r1, _ = env1.step(a2)
        # Now r1, r2 are the true payoffs in {+1, -1}

        # ------------ UPDATES for BOTH ------------
        # Convert reward ∈ {+1, -1} → loss ∈ {0, 1}
        loss1 = (1 - r1) / 2.0   # either 0 or 1
        loss2 = (1 - r2) / 2.0

        # Hedge update at (h=0, s=0) for each learner
        learner1.bandits[0][s1].update(a1, loss1)
        learner2.bandits[0][s2].update(a2, loss2)

        # Manually increment N_count and store that episode’s distribution
        # (since VLearning no longer does this internally for “manual” runs)
        learner1.N_count[0, s1] += 1
        i1 = learner1.N_count[0, s1]
        learner1.stored_distributions[(0, s1, i1)] = dist1.copy()

        learner2.N_count[0, s2] += 1
        i2 = learner2.N_count[0, s2]
        learner2.stored_distributions[(0, s2, i2)] = dist2.copy()

        # End of one “episode” for both players

    # After K episodes, extract each player’s final mixed policy
    # get_output_policy() will read learner.N_count and learner.stored_distributions
    policy1 = learner1.get_output_policy()  # dict: policy1[h][s]
    policy2 = learner2.get_output_policy()

    # For Matching Pennies, h=0 and s=0 is the only state
    return policy1[0][0], policy2[0][0]


if __name__ == "__main__":
    K = 5000
    p1_mixture, p2_mixture = run_two_player_vlearning(K=K, seed=123)
    print(f"After K={K} episodes using full V-Learning (mixture):")
    print("  P1’s final mixed policy:", p1_mixture)
    print("  P2’s final mixed policy:", p2_mixture)
