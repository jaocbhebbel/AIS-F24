# Othello Project

This othello AI is a project done by Jacob & Ricardo. We are going to use a reinforcement learning approach to training our neural network, and deploying this model inside a pygame environment of the board game.


## Papers we have looked at so far:
- [Application of Reinforcement Learning to the Game Othello](https://www.sciencedirect.com/science/article/pii/S0305054806002553#:~:text=We%20describe%20how%20reinforcement%20learning%20can%20be%20combined,state%20space%E2%80%94learning%20to%20play%20the%20game%20of%20Othello)
- [Reinforcement Learning in the Game of Othello: Learning Against a Fixed Opponent and Learning from Self-Play](https://www.ai.rug.nl/%7Emwiering/GROUP/ARTICLES/paper-othello.pdf)

## Useful articles / videos:
- [What is DQN?](https://medium.com/data-science-in-your-pocket/deep-q-networks-dqn-explained-with-examples-and-codes-in-reinforcement-learning-928b97efa792)
- [Explanantion of Monte Carlo](https://www.youtube.com/watch?v=ZljeS5aHzuE)

## Procedure for building a reinforcement learning algorithm for Othello:

### 1. define an environment for our agent to explore

An environment is the place where various states of the game exist. Our model will analyze potential, legal states to decide which move is best to play. 

This consists of building the board game in python as a 2 dimensional collection of cells in various states. The three states our model will have to understand are black, white, and empty. It can process these states as integers -1, 0, 1. This can be represented in a visual medium to a client as a board built in pygame.

The various actions a model can take in our environment consists of all legal moves for a player's turn. In Othello, a legal move is defined as a move that flips at least one of the opponent's pieces over. If this cannot be achieved, control is passed to the opponent without the player making a move.

Our reward function should consider a variety of statistics, such as # of pieces flipped, controlling edges/corners, and mobility of future moves. There are a variety of implementations, such as decision trees that look at future turns with the highest reward value to make the best move that predicts your opponent's move. Part of our research should include understanding the different reward functions.

### 2. Choosing a reinforcement learning algorithm

There are quite a few reinforcement learning algorithms out there. The