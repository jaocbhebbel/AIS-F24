BLACK, EMPTY, WHITE = -1, 0, 1
board = [[EMPTY] * 8] * 8
for row in board:
    print (row)

def printBoard(board):
    for row in board:
        for square in row:
            if square == BLACK:
                print("B", end="\t")
            elif square == WHITE:
                print("W", end="\t")
            elif square == EMPTY:
                print("O", end="\t")
            
        print("\n")



'''
validate function:
verifies a move is proper chess notation (letter-number)
letter must be a - h
number must be 1 - 8
'''

def validate(move):
    if len(move) != 2:
        return False
    if ord(move.substr(0,1)) - 97 < 0 or ord(move.substr(0,1)) - 97 > 8:
        return False
    if int(move.substr(1,2), 10) < 0 or int(move.substr(1,2), 10) > 8:
        return False
    
    return True


def toGrid(move):
    return (ord(move.substr(0,1)) - 97, ord(move.substr(1,2)) - 1)

def move(turn):
    if turn == BLACK:
        move = input("Black, enter the chess notation location of your move:\t\t").strip().lower()
        while validate(move) != True:
            move = input("You entered:\t\t", move, "\nBlack, enter the chess notation location of your move:\t\t").strip().lower()
        move = toGrid(move)
        board[move[0]][move[1]] = BLACK
        print(chr(27) + "[2J")
    
    else:
        move = input("White, enter the chess notation location of your move:\t\t").strip().lower()
        while validate(move) != True:
            move = input("You entered:\t\t", move, "\nWhite, enter the chess notation location of your move:\t\t").strip().lower()
        move = toGrid(move)
        board[move[0]][move[1]] = WHITE
        print(chr(27) + "[2J")

    return
    
    

def checkWin(board):
    for row in board:
        for square in row:
            if square == EMPTY:
                return False
            
    
    return True

def main():
    print("This is the othello game. Place a piece using chess notation (rows A-H, cols 1-8)")

    board[3][3] = BLACK
    board[3][4] = BLACK
    board[3][4] = WHITE
    board[4][3] = WHITE


    turn = 0

    while turn != BLACK and turn != WHITE:
        
        turn = input("who is going first?\t\t").strip().lower()
        print(turn)

        if turn.strip().lower() == 'white':
            turn = WHITE
            print("valid A")
            print(turn)
        elif turn == 'black':
            turn = BLACK
            print("valid B")
        else:
            input("please enter \'white\' or \'black\' to select who is going first")
    
    while checkWin(board) != True:
        print("The board:\n\n")
        printBoard(board)
        move(turn)
        turn *= -1


main()    

