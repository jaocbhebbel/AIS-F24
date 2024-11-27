import torch
import torch.nn as nn
import torch.optim as optim

class TicTacToeNN(nn.Module):
    def __init__(self):
        super(TicTacToeNN, self).__init__()
        self.fc1 = nn.Linear(9, 16)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(16, 16)  # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(16, 9)  # Second hidden layer to output layer
        self.relu = nn.ReLU()  # Activation function
        self.softmax = nn.Softmax(dim=1)  # Softmax for output layer

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Input to hidden layer 1
        x = self.relu(self.fc2(x))  # Hidden layer 1 to hidden layer 2
        x = self.softmax(self.fc3(x))  # Hidden layer 2 to output layer
        return x

# Initialize the model
model = TicTacToeNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example training loop
def train(model, optimizer, criterion, epochs, data_loader):
    model.train()
    for epoch in range(epochs):
        for board_states, labels in data_loader:  # Assuming data_loader provides (input, target)
            optimizer.zero_grad()  # Zero the gradient buffers
            outputs = model(board_states)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Example usage
# You need to define a DataLoader (data_loader) providing board states and labels
# train(model, optimizer, criterion, epochs=10, data_loader=data_loader)
