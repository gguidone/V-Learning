# =====================================
# run_tictactoe.py
# =====================================
import numpy as np
from tqdm import trange
from tictactoe.tictactoe_mdp import TicTacToeMDP
from vlearning.vlearning import VLearning

def run_two_player_tictactoe(K=20000, seed=None):
    """
    Run two V-Learning agents (X as player_id=1, O as player_id=2) on Tic-Tac-Toe.
    We wrap the outer loop in `trange` so that tqdm shows progress.
    Returns each player’s final mixed policy at each (h, s).
    """

    if seed is not None:
        np.random.seed(seed)

    # Create two MDP instances, one for each player
    env1 = TicTacToeMDP(player_id=1)  # X’s environment
    env2 = TicTacToeMDP(player_id=2)  # O’s environment

    learner1 = VLearning(env=env1, num_episodes=K)
    learner2 = VLearning(env=env2, num_episodes=K)

    # Wrap the episode loop in trange(...)
    for episode in trange(1, K + 1, desc="Training Episodes"):
        # Reset both envs (empty board, X’s turn)
        s1 = env1.reset()  
        s2 = env2.reset()  

        # Up to H=9 steps or until someone wins/draws
        for h in range(env1.H):
            turn = env1.turn  # Whose turn it is (1 or 2)

            if turn == 1:
                # ---- X’s move ----
                dist1 = learner1.bandits[h][s1].get_distribution()
                a1 = np.random.choice(learner1.A, p=dist1)

                # Store for mixture:
                visit_i_1 = learner1.N_count[h, s1] + 1
                learner1.stored_distributions[(h, s1, visit_i_1)] = dist1.copy()

                # Take X’s step in env1:
                s1_next, r1, done1 = env1.step(a1)

                # SYNC O’s MDP (env2) to match env1’s board & turn
                env2.board      = env1.board.copy()
                env2.turn       = env1.turn
                env2.move_count = env1.move_count
                s2_next = env2._encode_state()

                # Update X’s V and Hedge at (h, s1):
                V1_next = 0.0 if done1 else learner1.V[h + 1, s1_next]
                loss1 = (learner1.H - (r1 + V1_next)) / float(learner1.H)
                loss1 = np.clip(loss1, 0.0, 1.0)

                learner1.bandits[h][s1].update(a1, loss1)
                learner1.N_count[h, s1] += 1

                alpha1 = (learner1.H + 1) / float(learner1.H + learner1.N_count[h, s1])
                learner1.V_tilde[h, s1] = (
                    (1 - alpha1) * learner1.V_tilde[h, s1]
                    + alpha1 * (r1 + V1_next)
                )
                learner1.V[h, s1] = min(learner1.H + 1 - h, learner1.V_tilde[h, s1])

                # Advance state codes
                s1 = s1_next
                s2 = s2_next

                if done1:
                    # X just won (r1=+1) or made illegal move (r1=-1)
                    break

            else:
                # ---- O’s move ----
                dist2 = learner2.bandits[h][s2].get_distribution()
                a2 = np.random.choice(learner2.A, p=dist2)

                visit_i_2 = learner2.N_count[h, s2] + 1
                learner2.stored_distributions[(h, s2, visit_i_2)] = dist2.copy()

                # Take O’s step in env2:
                s2_next, r2, done2 = env2.step(a2)

                # SYNC X’s MDP (env1) to match env2’s board & turn
                env1.board      = env2.board.copy()
                env1.turn       = env2.turn
                env1.move_count = env2.move_count
                s1_next = env1._encode_state()

                # Update O’s V and Hedge at (h, s2):
                V2_next = 0.0 if done2 else learner2.V[h + 1, s2_next]
                loss2 = (learner2.H - (r2 + V2_next)) / float(learner2.H)
                loss2 = np.clip(loss2, 0.0, 1.0)

                learner2.bandits[h][s2].update(a2, loss2)
                learner2.N_count[h, s2] += 1

                alpha2 = (learner2.H + 1) / float(learner2.H + learner2.N_count[h, s2])
                learner2.V_tilde[h, s2] = (
                    (1 - alpha2) * learner2.V_tilde[h, s2]
                    + alpha2 * (r2 + V2_next)
                )
                learner2.V[h, s2] = min(learner2.H + 1 - h, learner2.V_tilde[h, s2])

                # Advance state codes
                s1 = s1_next
                s2 = s2_next

                if done2:
                    # O just won (r2=+1) or made illegal move (r2=-1)
                    break

        # End of this one TicTacToe episode

    # After K episodes, extract each player's final mixed policy
    policy1 = learner1.get_output_policy_sparse()
    policy2 = learner2.get_output_policy_sparse()
    return policy1, policy2


if __name__ == "__main__":
    K = 20000
    p1_policy, p2_policy = run_two_player_tictactoe(K=K, seed=2025)

    # Print X’s mixed policy at h=0, s=0 (empty board).
    dist1_h0 = p1_policy[0][0]
    print("P1 (X) mixed policy at h=0, s=empty:", dist1_h0)
