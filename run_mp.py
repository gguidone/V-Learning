import numpy as np
from bandit_with_decay import HedgeBandit  # uses η_t = sqrt((H ln B)/(B t))
from matching_pennies import MatchingPennies

def run_matching_pennies(K=50000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    B = 2   # two actions: Heads or Tails
    H = 1   # horizon = 1 for Matching Pennies

    bandit1 = HedgeBandit(num_actions=B, horizon_H=H)
    bandit2 = HedgeBandit(num_actions=B, horizon_H=H)
    env = MatchingPennies()

    sum_dist1 = np.zeros(B)
    sum_dist2 = np.zeros(B)

    for t in range(1, K+1):
        # 1) Each player samples from its Hedge bandit
        dist1 = bandit1.get_distribution()
        dist2 = bandit2.get_distribution()

        # 2) Record for time‐average
        sum_dist1 += dist1
        sum_dist2 += dist2

        a1 = np.random.choice(B, p=dist1)
        a2 = np.random.choice(B, p=dist2)

        # 3) Environment payoff
        r1, r2 = env.step(a1, a2)

        # 4) Convert to [0,1] loss
        loss1 = (1 - r1) / 2
        loss2 = (1 - r2) / 2

        # 5) Update each Hedge bandit (this uses η_t = sqrt((H ln B)/(B t)))
        bandit1.update(a1, loss1)
        bandit2.update(a2, loss2)

    # Compute time‐averaged distributions
    avg_dist1 = sum_dist1 / K
    avg_dist2 = sum_dist2 / K

    # Compute each player’s final single‐iterate dist
    final_dist1 = bandit1.get_distribution()
    final_dist2 = bandit2.get_distribution()

    return avg_dist1, avg_dist2, final_dist1, final_dist2

if __name__ == "__main__":
    K = 50000
    avg1, avg2, final1, final2 = run_matching_pennies(K=K, seed=42)

    print(f"After K={K} episodes:")
    print("  Time‐averaged P1:", avg1)
    print("  Time‐averaged P2:", avg2)
    print("  Final single‐iterate P1:", final1)
    print("  Final single‐iterate P2:", final2)
