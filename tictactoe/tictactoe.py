"""
Tic Tac Toe Player
"""

import math
import copy
import numpy as np


X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """

    # flatten board and count
    board_flat = [field for row in board for field in row]
    turn_difference = board_flat.count(X) - board_flat.count(O)

    if turn_difference == 0:
        return X
    elif turn_difference == 1:
        return O
    else:
        raise Exception("Invalid board state detected")


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    actions_set = set()

    # i = row, j = column
    for i, row in enumerate(board):
        for j, field in enumerate(row):
            if field == EMPTY:
                actions_set.add((i, j))

    return actions_set


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    # Determine target field and current player
    r, c = action

    # Check coordinates
    try:
        target = board[r][c]
    except IndexError:
        raise Exception("Chosen coordinates ROW:{} COL:{} do not exist".format(r, c))

    # Check target value
    if target is not EMPTY:
        raise Exception("Target field is not empty. Current value is:{}".format(target))

    # Create deep copy and modify the target field to the current player value
    board_updated = copy.deepcopy(board)
    board_updated[r][c] = player(board)

    return board_updated


def check_win(three_values):
    v1, v2, v3 = three_values
    if v1 == v2 == v3 and v1 is not EMPTY:
        return v1
    else:
        return None


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    # get diagonals, columns and rows
    combinations = [(board[0][0], board[1][1], board[2][2]), # diagonal top left to lower right
                    (board[2][0], board[1][1], board[0][2])] # diagonal lower left to upper right
    for i in range(3):
        combinations.append((board[0][i], board[1][i], board[2][i]))
        combinations.append((board[i][0], board[i][1], board[i][2]))

    # check each combination for a win
    for three_fields in combinations:
        foo = check_win(three_fields)
        if foo:
            return foo

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    # Check if all fields are taken:
    board_flat = [field for row in board for field in row]
    if board_flat.count(EMPTY) == 0:
        return True

    # Check for winning state
    if winner(board):
        return True

    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    w = winner(board)

    if w == X:
        return 1
    elif w == O:
        return -1
    else:
        return 0


def maxvalue(board):

    if terminal(board):
        return utility(board)

    v = -np.inf
    for action in actions(board):
        v = max(v, minvalue(result(board, action)))
    return v


def minvalue(board):
    if terminal(board):
        return utility(board)

    v = np.inf
    for action in actions(board):
        v = min(v, maxvalue(result(board, action)))
    return v





def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    # Check terminal condition
    if terminal(board):
        return None

    # get all actions and make them indexable
    all_actions = list(actions(board))

    # get the best index (for the actions) accordingly to the current player
    if player(board) == X:
        # Maximizing Player
        values = [minvalue(result(board, a)) for a in all_actions]
        best_index = values.index(max(values))
    else:
        # Minimizing Player
        values = [maxvalue(result(board, a)) for a in all_actions]
        best_index = values.index(min(values))

    return all_actions[best_index]
