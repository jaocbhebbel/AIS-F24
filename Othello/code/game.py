import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
BOARD_SIZE = 8  # 8x8 board
CELL_SIZE = 80  # Cell size in pixels
SCREEN_SIZE = BOARD_SIZE * CELL_SIZE
WHITE, BLACK, EMPTY, HIGHLIGHT = 1, -1, 0, 2

# Colors
GREEN = (34, 139, 34)
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
HIGHLIGHT_COLOR = (255, 215, 0)

# Initialize screen
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Othello")

# Initialize an empty 8x8 board
board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

# Place initial pieces
board[3][3] = WHITE
board[3][4] = BLACK
board[4][3] = BLACK
board[4][4] = WHITE

def draw_board():
    screen.fill(GREEN)
    # Draw grid
    for x in range(1, BOARD_SIZE):
        pygame.draw.line(screen, BLACK_COLOR, (x * CELL_SIZE, 0), (x * CELL_SIZE, SCREEN_SIZE))
        pygame.draw.line(screen, BLACK_COLOR, (0, x * CELL_SIZE), (SCREEN_SIZE, x * CELL_SIZE))

    # Draw pieces
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == WHITE:
                pygame.draw.circle(screen, WHITE_COLOR, (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2 - 5)
            elif board[row][col] == BLACK:
                pygame.draw.circle(screen, BLACK_COLOR, (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2 - 5)

def is_valid_move(board, row, col, color):
    if board[row][col] != EMPTY:
        return False
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    for dr, dc in directions:
        r, c = row + dr, col + dc
        count = 0
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == -color:
            r += dr
            c += dc
            count += 1
        if count > 0 and 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == color:
            return True
    return False

def place_piece(board, row, col, color):
    if not is_valid_move(board, row, col, color):
        return False
    board[row][col] = color
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    for dr, dc in directions:
        r, c = row + dr, col + dc
        flip_positions = []
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == -color:
            flip_positions.append((r, c))
            r += dr
            c += dc
        if flip_positions and 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == color:
            for fr, fc in flip_positions:
                board[fr][fc] = color
    return True

def get_score(board):
    white_score = np.sum(board == WHITE)
    black_score = np.sum(board == BLACK)
    return white_score, black_score

def is_game_over(board):
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == EMPTY and (is_valid_move(board, row, col, BLACK) or is_valid_move(board, row, col, WHITE)):
                return False
    return True

def main():
    clock = pygame.time.Clock()
    current_color = BLACK
    running = True

    while running:
        draw_board()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                row, col = y // CELL_SIZE, x // CELL_SIZE
                if is_valid_move(board, row, col, current_color):
                    place_piece(board, row, col, current_color)
                    current_color = -current_color  # Switch turns

                    if is_game_over(board):
                        running = False  # Exit the loop to end the game
                        white_score, black_score = get_score(board)
                        print("Game Over")
                        print(f"White: {white_score}, Black: {black_score}")
                        break  # Exit the for-loop if the game is over

        clock.tick(30)
    pygame.quit()

if __name__ == "__main__":
    main()