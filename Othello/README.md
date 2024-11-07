# Othello Project

This othello AI is a project done by Jacob & Ricardo. We are going to use a reinforcement learning approach to training our neural network, and deploying this model inside a pygame environment of the board game.


## Papers we have looked at so far:
- [Application of Reinforcement Learning to the Game Othello](https://www.sciencedirect.com/science/article/pii/S0305054806002553#:~:text=We%20describe%20how%20reinforcement%20learning%20can%20be%20combined,state%20space%E2%80%94learning%20to%20play%20the%20game%20of%20Othello)
- [Reinforcement Learning in the Game of Othello: Learning Against a Fixed Opponent and Learning from Self-Play](https://www.ai.rug.nl/%7Emwiering/GROUP/ARTICLES/paper-othello.pdf)


## Procedure for building a reinforcement learning algorithm for Othello:

### 1. define an environment for our agent to explore

This consists of building the board game in python as a 2 dimensional collection of cells in various states. The three states our model will have to understand are black, white, and empty. It can process these states as integers -1, 0, 1. This can be represented in a visual medium to a client as a board built in pygame.

### 2. define the various actions a model can take in this environment

The various actions a model can take in our environment. This consists of all legal moves for a player's turn. We need to translate 