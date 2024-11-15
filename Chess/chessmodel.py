import random
import chess
import chess.engine
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

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
def encode_board(board){
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
    
    # After encoding, return planes with unsqueeze (which adds a dimension to the planes tensor)
    return planes.unsqueeze(0)
}

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

#Main Function
if __name__ == "__main__":
    #Define Board, depth and model
    board = chess.Board()
    maxdepth = 18
    model = ChessCNN()

    generate_board(board, maxdepth)
    white_material, black_material = calculate_material(board)
    print(white_material)
    print(black_material)
