from itertools import product

def is_winner(board, player):
    """Checks if the given player has won on the board."""
    win_conditions = [
        # Rows
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        # Columns
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        # Diagonals
        [0, 4, 8], [2, 4, 6]
    ]
    return any(all(board[i] == player for i in condition) for condition in win_conditions)

def is_valid_board(board):
    """Checks if the board is a valid Tic-Tac-Toe state."""
    x_count = board.count("X")
    o_count = board.count("O")

    # Rule 1: X must be equal to or one more than O
    if not (x_count == o_count or x_count == o_count + 1):
        return False

    # Rule 2: Both players cannot win simultaneously
    x_wins = is_winner(board, "X")
    o_wins = is_winner(board, "O")
    if x_wins and o_wins:
        return False

    # Rule 3: If X wins, X must have one more move than O
    if x_wins and x_count != o_count + 1:
        return False

    # Rule 4: If O wins, X and O must have the same number of moves
    if o_wins and x_count != o_count:
        return False

    return True

def generate_valid_boards():
    """Generates and prints all valid Tic-Tac-Toe boards."""
    symbols = [" ", "X", "O"]
    all_boards = product(symbols, repeat=9)

    valid_boards = []
    for board in all_boards:
        if is_valid_board(board):
            valid_boards.append(board)

    # Print the valid boards
    for idx, board in enumerate(valid_boards):
        print(f"Valid Board #{idx + 1}")
        for i in range(0, 9, 3):
            print(f"{board[i]} | {board[i+1]} | {board[i+2]}")
            if i < 6:
                print("--+---+--")
        print("\n")

    print(f"Total valid boards: {len(valid_boards)}")

# Run the function
generate_valid_boards()
