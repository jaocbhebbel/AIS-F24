import random
import chess
import chess.engine
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms

# define piece images
PIECE_IMAGES = {
    'p': 'blackpawn.png',
    'n': 'blackknight.png',
    'b': 'blackbishop.png',
    'r': 'blackrook.png',
    'q': 'blackqueen.png',
    'k': 'blackking.png',
    'P': 'whitepawn.png',
    'N': 'whiteknight.png',
    'B': 'whitebishop.png',
    'R': 'whiterook.png',
    'Q': 'whitequeen.png',
    'K': 'whiteking.png',
}

# initialize pygame
pygame.init()

# set up display dimensions
WIDTH, HEIGHT = 640, 512
SQUARE_SIZE  = WIDTH // 8
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ThunderByte Chess")

# colors
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)

class ThunderByteCNN(nn.Module):
    #Initialize the Convolutional Neural Network
    def __init__(self):
        #Calls the parent function (my CNN model) with the parameters and abilities of the nn.Module from pytorch
        super(ThunderByteCNN, self).__init__()

        #Create convolutional layers
        self.convlayer1 = nn.Conv2d(14, 32, kernel_size=3, padding=1)
        self.convlayer2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.convlayer3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Define fully connected layers for final output
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1) #Output will be eval (between 0/1 tanh bounds)

    #Forward (Forward Pass):
    #Describes how to put each layer to input and what to output using (tanh (-1/1 output)
    def forward(self):
        x = torch.relu(self.convlayer1(x))
        x = torch.relu(self.convlayer2(x))
        x = torch.relu(self.convlayer3(x))
        x = x.view(-1, 128 * 8 * 8)  # Flatten for fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # Output between -1 and 1
        return x

#Monte Carlo Tree Search Algorithm for Model
#Allows the CNN to explore different random moves and figure out how good a move is (q values)
#Makes the CNN play better in simple terms
class MCTS:
    #Initialize the model qith q values and visit counts
    def __init__(self, model, simulations = 1000):
        self.model = model
        self.simulations = simulations
        self.q_values = {}  # Stores Q values for each board position
        self.visit_counts = {}  # Tracks visit counts for each board position
    
    #Function to select amove based on the board
    def select_move(self, board):
        # Select the best move based on visit counts
        legal_moves = list(board.legal_moves)
        move_scores = {}
        for move in legal_moves:
            board.push(move)
            move_scores[move] = self.q_values.get(board.fen(), 0) / self.visit_counts.get(board.fen(), 1)
            board.pop()
        
        best_move = max(move_scores, key=move_scores.get)
        return best_move

#Function to generate a random board for neural network to analyze
def generate_board(board, depth):
    if (depth == 0): 
        return board
    else:
        move = random.choice(list(board.legal_moves))
        board.push(move)
        print()
        print("Move:", move)
        print(board)
        generate_board(board, depth - 1)

def draw_board(window, board):
    """
    Draws the chessboard and pieces onto the Pygame window.
    
    Parameters:
        window (pygame.Surface): The Pygame window to draw on.
        board (chess.Board): The chess board state from python-chess.
    """
    # Loop through all squares (0 to 63)
    for rank in range(8):
        for file in range(8):
            square = rank * 8 + file
            
            # Determine the color of the square
            is_light_square = (rank + file) % 2 == 0
            color = LIGHT_SQUARE if is_light_square else DARK_SQUARE
            
            # Draw the square
            pygame.draw.rect(
                window,
                color,
                pygame.Rect(file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            )
            
            # Draw the piece if one is on the square
            piece = board.piece_at(square)
            if piece:
                piece_image = PIECE_IMAGES.get(piece.symbol())
                if piece_image:
                    window.blit(
                        piece_image,
                        (file * SQUARE_SIZE, rank * SQUARE_SIZE)
                    )

def draw_board_with_panel(window, board, player_input):
    """
    Draw the chessboard and the side panel for player input.
    
    Parameters:
        window (pygame.Surface): The Pygame window to draw on.
        board (chess.Board): The chess board state from python-chess.
        player_input (str): Text input from the player for moves.
    """
    # Draw the chessboard
    for rank in range(8):
        for file in range(8):
            square_color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(window, square_color, (file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            
            # Draw pieces
            piece = board.piece_at(chess.square(file, 7 - rank))  # Convert to 0-based indexing
            if piece:
                piece_image = PIECE_IMAGES[piece.symbol()]
                window.blit(piece_image, (file * SQUARE_SIZE, rank * SQUARE_SIZE))
    
    # Draw the side panel
    pygame.draw.rect(window, (50, 50, 50), (WIDTH - 128, 0, 128, HEIGHT))  # Background for panel
    font = pygame.font.Font(None, 36)
    text = font.render("Player Move:", True, (255, 255, 255))
    window.blit(text, (WIDTH - 120, 20))
    
    # Render player input
    input_text = font.render(player_input, True, (200, 200, 200))
    window.blit(input_text, (WIDTH - 120, 60))

def handle_player_move(board, move_input):
    """
    Processes a move input by the player and updates the board.
    
    Parameters:
        board (chess.Board): The current chess board state.
        move_input (str): Player's move in UCI or algebraic notation.
    
    Returns:
        bool: True if the move is valid, False otherwise.
    """
    try:
        move = chess.Move.from_uci(move_input)
        if move in board.legal_moves:
            board.push(move)
            return True
        else:
            print("Illegal move!")
            return False
    except ValueError:
        print("Invalid move format!")
        return False


#Function to encode the different piece boards with 0s and 1s
def encode_board(board):
    # 14 planes, 8x8 board
    # 12 planes for each piece on the board
    # 1 board for current turn (all 0s is black, 1s is white)
    # 1 board for castling rights/rules etc...
    planes = torch.zeros((14, 8, 8))

    #Loop through each square, check for pieces at each square, encode the planes with 0s and 1s
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
            planes[plane, square // 8, square % 8] = 1
        else:
            # Optional: You can add a comment or logic here for clarity
            pass  # No piece on this square, leave it as zeros
    # After encoding, return planes with unsqueeze (which adds a dimension to the planes tensor)
    return planes.unsqueeze(0)


#Determine the Material on both sides 
def calculate_material(board):
    # Define the piece values
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }

    # Initialize the material score
    white_material = 0
    black_material = 0

    # Iterate over all pieces on the board
    for square, piece in board.piece_map().items():
        # Get the value of the piece type (e.g., PAWN, KNIGHT, etc.)
        piece_value = piece_values.get(piece.piece_type, 0)

        # Add to the appropriate color's total
        if piece.color == chess.WHITE:
            white_material += piece_value
        else:
            black_material += piece_value

    return white_material, black_material

# load piece images
def load_images():
    for piece, filename in PIECE_IMAGES.items():
        try:
            # Load the image file associated with the piece
            image = pygame.image.load(f'./assets/{filename}')
            # Scale the image to fit the square size
            image = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))
            # Update the dictionary with the Pygame surface
            PIECE_IMAGES[piece] = image
        except FileNotFoundError:
            print(f"Error: Image file for {piece} not found at './assets/{filename}'!")

player_input = ""

#Main Function
if __name__ == "__main__":
    #Define Board, depth and model
    board = chess.Board()
    maxdepth = 18
    model = ThunderByteCNN()
    load_images()

    # Check if assets folder exists
    if not os.path.exists('assets'):
        print("Error: 'assets' folder not found!")
    else:
        print("'assets' folder is accessible!")

    # Initialize the board and load images
    board = chess.Board()
    load_images()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if handle_player_move(board, player_input):
                        player_input = ""  # Clear input on valid move
                    else:
                        print("Invalid or illegal move.")
                elif event.key == pygame.K_BACKSPACE:
                    player_input = player_input[:-1]  # Remove last character
                else:
                    player_input += event.unicode  # Add typed character to input
        
        # Redraw the board and side panel
        draw_board_with_panel(WINDOW, board, player_input)
        pygame.display.flip()
    
    pygame.quit()

    pygame.quit()