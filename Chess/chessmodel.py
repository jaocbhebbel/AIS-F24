import random
import chess
import chess.engine
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import assets
from torchvision import datasets, transforms

# define piece images
PIECE_IMAGES = {
    'p': 'black_pawn.png',
    'n': 'black_knight.png',
    'b': 'black_bishop.png',
    'r': 'black_rook.png',
    'q': 'black_queen.png',
    'k': 'black_king.png',
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
        # Run MCTS simulations
        for _ in range(self.simulations):
            self.simulate(board)

        # Select the best move based on visit counts
        legal_moves = list(board.legal_moves)
        move_scores = {}
        for move in legal_moves:
            board.push(move)
            move_scores[move] = self.q_values.get(board.fen(), 0) / self.visit_counts.get(board.fen(), 1)
            board.pop()
        
        best_move = max(move_scores, key=move_scores.get)
        return best_move

    #Simulate a game
    def simulate(self, board):
        if board.is_game_over():
            return self.evaluate_terminal(board)
        
        # Initialize Q and N for a new position
        board_fen = board.fen()
        if board_fen not in self.q_values:
            self.q_values[board_fen] = 0
            self.visit_counts[board_fen] = 0
            return self.evaluate(board)
        
        # Select a random legal move
        legal_moves = list(board.legal_moves)
        move = random.choice(legal_moves)
        board.push(move)
        reward = -self.simulate(board)
        board.pop()

        # Update Q and N values
        self.q_values[board_fen] += reward
        self.visit_counts[board_fen] += 1
        return reward

    #Evaluate the model based on material and nn eval
    def evaluate(self, board):
        # Encode the board for the model
        board_tensor = encode_board(board)
        with torch.no_grad():
            nn_eval = self.model(board_tensor).item()
        
        # Blend the neural network evaluation with material evaluation
        white_material, black_material = calculate_material(board)
        material_eval = (white_material - black_material) / 10.0  # Normalize material scores
        return nn_eval * 0.7 + material_eval * 0.3  # Weighted average

    #Evaluate with tanh (-1/1)
    def evaluate_terminal(self, board):
        # Terminal evaluation based on game outcome
        if board.is_checkmate():
            return -1  # Loss
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0  # Draw
        return 1  # Win

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

#Self play function
def self_play(model, iterations=10):
    for _ in range(iterations):
        board = chess.Board()
        mcts = MCTS(model)
        
        while not board.is_game_over():
            move = mcts.select_move(board)
            board.push(move)

        # Evaluate game result and update model weights here if using reinforcement learning
        print(f"Game result: {board.result()}")

# load piece images
def load_images():
    pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    for piece in pieces:
        image = pygame.image.load(f'assets/black_pawn.png')
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
