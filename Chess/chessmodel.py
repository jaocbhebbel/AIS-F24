import random
import chess
import chess.engine
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# define piece images
PIECE_IMAGES = {
    'p': 'black_pawn.png',
    'n': 'black_knight.png',
    'b': 'black_bishop.png',
    'r': 'black_rook.png',
    'q': 'black_queen.png',
    'k': 'black_king.png'
    'P': 'white_pawn.png',
    'N': 'white_knight.png',
    'B': 'white_bishop.png',
    'R': 'white_rook.png',
    'Q': 'white_queen.png',
    'K': 'white_king.png',
}

# initialize pygame
pygame.init()

# set up display dimensions
WIDTH, HEIGHT = 512, 512
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


    #def forward(self):
        
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
    pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    for piece in pieces:
        image = pygame.image.load(f'assets/pawn.png')
        image = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))
        PIECE_IMAGES[piece] = image

#Main Function
if __name__ == "__main__":
    #Define Board, depth and model
    board = chess.Board()
    maxdepth = 18
    model = ThunderByteCNN()
    load_images()

    generate_board(board, maxdepth)
    white_material, black_material = calculate_material(board)
    print(white_material)
    print(black_material)
