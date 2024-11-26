import tkinter as tk
from tkinter import messagebox
from NewCombos import generate_all_unique_valid_states, check_winner

class TicTacToeGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Tic Tac Toe")

        self.board = [0] * 9
        self.current_turn = 1  # X starts first

        self.buttons = []
        for i in range(9):
            button = tk.Button(self.window, text="", font=("Arial", 24), height=2, width=5,
                               command=lambda idx=i: self.make_move(idx))
            button.grid(row=i // 3, column=i % 3)
            self.buttons.append(button)

        reset_button = tk.Button(self.window, text="Reset", font=("Arial", 16), command=self.reset_board)
        reset_button.grid(row=3, column=0, columnspan=3, sticky="nsew")

        self.valid_states = generate_all_unique_valid_states()

    def make_move(self, idx):
        if self.board[idx] != 0:
            messagebox.showinfo("Invalid Move", "That space is already taken!")
            return

        self.board[idx] = self.current_turn
        self.buttons[idx].config(text="X" if self.current_turn == 1 else "O")

        winner = check_winner(self.board)
        if winner:
            self.display_winner(winner)
        elif all(space != 0 for space in self.board):
            messagebox.showinfo("Game Over", "It's a draw!")
            self.reset_board()
        else:
            self.current_turn = 2 if self.current_turn == 1 else 1

    def display_winner(self, winner):
        winner_text = "X" if winner == 1 else "O"
        messagebox.showinfo("Game Over", f"{winner_text} wins!")
        self.reset_board()

    def reset_board(self):
        self.board = [0] * 9
        self.current_turn = 1
        for button in self.buttons:
            button.config(text="")

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    gui = TicTacToeGUI()
    gui.run()
