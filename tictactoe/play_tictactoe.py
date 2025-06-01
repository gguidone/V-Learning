# =====================================
# play_tictactoe.py
# =====================================

import numpy as np
import pickle
from tictactoe.tictactoe_mdp import TicTacToeMDP

def encode_state(board, turn):
    """
    Encode 3×3 board + turn into a single integer in [0, 2*3^9).
    board: list of 9 ints (0=empty,1=X,2=O).
    turn: 1 or 2.
    """
    code = 0
    for cell in board:
        code = code * 3 + cell
    if turn == 2:
        code += 3 ** 9
    return code

def is_winner(board, pid):
    """
    Check if player pid (1 or 2) has 3 in a row on the board.
    """
    wins = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]
    for (i,j,k) in wins:
        if board[i] == pid and board[j] == pid and board[k] == pid:
            return True
    return False

def print_board(board):
    """
    Nicely print the 3×3 board.
    """
    symbols = {0: ' ', 1: 'X', 2: 'O'}
    for r in range(3):
        row = ' | '.join(symbols[board[3*r + c]] for c in range(3))
        print(' ' + row)
        if r < 2:
            print('---+---+---')

def play_against_policy(policy_X, policy_O):
    """
    Let human play one Tic-Tac-Toe game against the V-Learning AI.
    policy_X and policy_O are dicts returned by get_output_policy(), 
    i.e. policy_[move_count][state_code] gives a length-9 distribution over moves.
    """
    choice = ''
    while choice not in ('X','O'):
        choice = input("Play as X (first) or O (second)? [X/O]: ").strip().upper()
    human = 1 if choice == 'X' else 2
    ai    = 2 if human == 1 else 1

    board = [0]*9
    turn = 1      # X always starts
    move_count = 0

    print("\nCell indices:\n 0 | 1 | 2\n---+---+---\n 3 | 4 | 5\n---+---+---\n 6 | 7 | 8\n")
    print("Let’s start!\n")

    while True:
        print_board(board)
        print()

        if turn == human:
            valid = False
            while not valid:
                try:
                    mv = int(input("Your move (0–8): "))
                    if 0 <= mv < 9 and board[mv] == 0:
                        valid = True
                    else:
                        print("Invalid or occupied. Try again.")
                except ValueError:
                    print("Enter a number 0–8.")
            board[mv] = human
            move_count += 1

            if is_winner(board, human):
                print_board(board)
                print("You win!\n")
                return
            if move_count == 9:
                print_board(board)
                print("Draw!\n")
                return

            turn = ai

        else:
            state_code = encode_state(board, turn)
            if turn == 1:
                dist = policy_X[move_count][state_code]
            else:
                dist = policy_O[move_count][state_code]
            mv = int(np.argmax(dist))
            print(f"AI ({'X' if turn==1 else 'O'}) chooses {mv}\n")
            board[mv] = turn
            move_count += 1

            if is_winner(board, turn):
                print_board(board)
                print("AI wins!\n")
                return
            if move_count == 9:
                print_board(board)
                print("Draw!\n")
                return

            turn = human

def main():
    # Load the saved policy files
    try:
        with open('policy_X.pkl', 'rb') as f:
            policy_X = pickle.load(f)
        with open('policy_O.pkl', 'rb') as f:
            policy_O = pickle.load(f)
    except FileNotFoundError:
        print("Missing policy_X.pkl or policy_O.pkl. Please run the training script first.")
        return

    play_against_policy(policy_X, policy_O)

if __name__ == "__main__":
    import pickle
    main()
