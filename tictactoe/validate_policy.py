# validate_policy.py

import pickle
import numpy as np

# 1) Helpers for encoding/decoding
def encode_state(board, turn):
    code = 0
    for cell in board:
        code = code * 3 + cell
    if turn == 2:
        code += 3**9
    return code

def print_top_moves(dist, topk=3):
    """
    Given a length-9 array dist, print the top-k moves (index + prob).
    """
    idxs = np.argsort(dist)[::-1][:topk]
    for i in idxs:
        print(f"  Move {i} → p = {dist[i]:.3f}")
    print()

# 2) Load the sparse policy dictionaries
with open('tictactoe/policy_X.pkl', 'rb') as f:
    policy_X = pickle.load(f)
with open('tictactoe/policy_O.pkl', 'rb') as f:
    policy_O = pickle.load(f)

uniform = np.ones(9) / 9.0

# 3) Define some “interesting” board states (list of length 9):
#   0=empty, 1=X, 2=O

# Example A: X to move at move_count=0 (empty board)
board_A = [0]*9
h_A = 0
state_A = encode_state(board_A, turn=1)
dist_A = policy_X.get(h_A, {}).get(state_A, uniform)
print("A) X’s opening distribution (h=0):")
print_top_moves(dist_A, topk=5)

# Example B: “X played center, now it’s O’s turn at h=1”
#   Board: [0,0,0,
#           0,1,0,
#           0,0,0], turn=2, h=1
board_B = [0,0,0,
           0,1,0,
           0,0,0]
h_B = 1
state_B = encode_state(board_B, turn=2)
dist_B = policy_O.get(h_B, {}).get(state_B, uniform)
print("B) O’s response after X took center (h=1):")
print_top_moves(dist_B, topk=5)

# Example C: “X: center & corner; O: opposite corner; X to move at h=2”
#   Board: [0,0,2,
#           0,1,0,
#           1,0,0], turn=1, h=2
board_C = [0,0,2,
           0,1,0,
           1,0,0]
h_C = 2
state_C = encode_state(board_C, turn=1)
dist_C = policy_X.get(h_C, {}).get(state_C, uniform)
print("C) X’s move, trying to create two-in‐a‐row (h=2):")
print_top_moves(dist_C, topk=5)

# Example D: “O has two‐in‐a‐row and must block”:
#   Assume X started center (1), O played corner (2), X played corner (h=2),
#   now O to move at h=3 with board like:
#     X | 0 | X
#     0 | O | 0
#     0 | 0 | 0
#   which is [1,0,1, 0,2,0, 0,0,0] and turn=2, h=3.
board_D = [1,0,1,
           0,2,0,
           0,0,0]
h_D = 3
state_D = encode_state(board_D, turn=2)
dist_D = policy_O.get(h_D, {}).get(state_D, uniform)
print("D) O must block X’s threat at (h=3):")
print_top_moves(dist_D, topk=5)
