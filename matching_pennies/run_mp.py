# =================================
# matching_pennies/run_mp.py
# =================================

import numpy as np

from matching_pennies.mp_mdp import MatchingPenniesMDP
from vlearning.vlearning import VLearning
from vlearning.ftrl_bandit import FTRLBandit

def run_two_player_vlearning(K=5000, seed=None):
    """
    Run two V-Learning agents on Matching Pennies (H=1).
    Each agent’s environment returns that agent’s own reward.
    We manually run K “episodes” in lockstep because H=1.
    Returns each player’s final mixed policy (a length-2 array).
    """
    if seed is not None:
        np.random.seed(seed)

    # Create two Matching Pennies MDPs, one for each player
    env1 = MatchingPenniesMDP(player_id=1)
    env2 = MatchingPenniesMDP(player_id=2)

    # Create two VLearning learners; each will lazily create its FTRLBandit at (h=0,s=0)
    learner1 = VLearning(env=env1, num_episodes=K)
    learner2 = VLearning(env=env2, num_episodes=K)

    # We manually run K “episodes.” Each episode is just h=0 for Matching Pennies.
    for episode in range(1, K + 1):
        # ---------------- PLAYER 1’s “episode” (h=0) ----------------
        s1 = env1.reset()  # always returns s1 = 0

        # Lazily create bandit for (h=0, s1) if needed
        if s1 not in learner1.bandits[0]:
            learner1.bandits[0][s1] = FTRLBandit(learner1.A, learner1.H, learner1.K)

        # Sample action via sample_action() (advances that bandit’s round)
        a1 = learner1.bandits[0][s1].sample_action()

        # Take the “dummy” step in P1’s MDP
        s1_next, r1_dummy, done1 = env1.step(a1)
        # (r1_dummy isn’t the final reward yet; we wait for P2)

        # ---------------- PLAYER 2’s “episode” (h=0) ----------------
        s2 = env2.reset()  # always returns s2 = 0

        # Lazily create bandit for (h=0, s2) if needed
        if s2 not in learner2.bandits[0]:
            learner2.bandits[0][s2] = FTRLBandit(learner2.A, learner2.H, learner2.K)

        # Sample action via sample_action() (advances that bandit’s round)
        a2 = learner2.bandits[0][s2].sample_action()

        # Now P2 steps; this resolves both players’ rewards
        s2_next, r2, done2 = env2.step(a2)
        _, r1, _ = env1.step(a2)  # P1’s true reward r1 ∈ {+1, -1}

        # ------------- CONSTRUCT LOSSES ∈ {0,1} -------------
        # In Matching Pennies, reward ∈ {+1, -1}; convert to loss ∈ {0,1}
        loss1 = (1 - r1) / 2.0
        loss2 = (1 - r2) / 2.0

        # ------------- UPDATE EACH BANDIT (h=0, s=0) -------------
        learner1.bandits[0][s1].update(a1, loss1)
        learner2.bandits[0][s2].update(a2, loss2)

        # ------------- MANUALLY TRACK V-LEARNING MIXTURE INFO -------------
        learner1.N_count[0, s1] += 1
        i1 = learner1.N_count[0, s1]
        dist1 = learner1.bandits[0][s1].last_distribution
        learner1.stored_distributions[(0, s1, i1)] = dist1.copy()

        learner2.N_count[0, s2] += 1
        i2 = learner2.N_count[0, s2]
        dist2 = learner2.bandits[0][s2].last_distribution
        learner2.stored_distributions[(0, s2, i2)] = dist2.copy()

        # End of one “episode” for both players

    # After K episodes, extract each player’s final mixed policy
    policy1 = learner1.get_output_policy()  # dict: policy1[h][s]
    policy2 = learner2.get_output_policy()

    # For Matching Pennies, (h=0, s=0) is the only relevant entry
    return policy1[0][0], policy2[0][0]


if __name__ == "__main__":
    K = 5000
    p1_mixture, p2_mixture = run_two_player_vlearning(K=K, seed=123)
    print(f"After K={K} episodes using full V-Learning (mixture):")
    print("  P1’s final mixed policy:", p1_mixture)
    print("  P2’s final mixed policy:", p2_mixture)
