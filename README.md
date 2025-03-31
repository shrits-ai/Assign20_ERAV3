# Self-Driving Car AI Project

## Overview

This project implements a self-driving car simulation using Kivy for the graphical user interface and a Deep Q-Network (DQN) for the AI. The car learns to navigate a custom-drawn track, following black lines and moving between specified goal points (A1, A2, A3). The AI is trained to maximize rewards, with positive rewards for staying on the track and reaching goals, and negative rewards for deviating from the track or hitting boundaries.

## Project Structure

The project consists of the following main components:

-   `map.py`: Contains the Kivy application, game logic, car movement, sensor handling, and reward system.
-   `ai.py`: Implements the Deep Q-Network (DQN) for the AI, including the neural network architecture, experience replay, and learning algorithm.
-   `images/`: Directory containing image assets, including the track map (`mask.png`, `MASK1.png`) and the dynamically generated sand map (`sand.jpg`).

## Dependencies

-   Python 3.x
-   Kivy
-   NumPy
-   Matplotlib
-   PIL (Pillow)
-   PyTorch

Install the dependencies using pip:

```bash
pip install kivy numpy matplotlib Pillow torch
```
## Running the Application

To run the self-driving car application, execute the `map.py` script:

```bash
python map.py
```
## AI Implementation (`ai.py`)

The AI is implemented using a Deep Q-Network (DQN) in `ai.py`. The DQN learns to approximate the Q-function, which estimates the optimal action to take in a given state.

### Neural Network Architecture

The neural network consists of two fully connected layers:

-   **Input Layer:** Takes the state as input (sensor readings and orientation).
-   **Hidden Layer:** 30 neurons with ReLU activation.
-   **Output Layer:** Outputs Q-values for each possible action (move straight, turn left, turn right).

### Experience Replay

Experience replay is used to stabilize the learning process by storing and sampling past experiences.

### DQN Algorithm

The DQN algorithm uses the following steps:

1.  **Select Action:** Choose an action using an epsilon-greedy strategy.
2.  **Observe Reward and Next State:** Receive a reward and observe the next state.
3.  **Store Experience:** Store the experience in the replay memory.
4.  **Learn:** Sample a batch of experiences from the replay memory and update the neural network weights.

### Saving and Loading the AI

The AI's model and optimizer state can be saved and loaded using the `save()` and `load()` methods.

## Game Logic (`map.py`)

The `map.py` script implements the game logic and user interface using Kivy.

### Car Movement

The car's movement is controlled by the AI's output, which determines the rotation angle. The car's velocity is adjusted based on whether it is on or off the black line.

### Sensor Handling

The car has three sensors that detect the presence of the black line. The sensor readings are used as input to the AI.

### Reward System

The reward system is designed to encourage the car to follow the black line and reach the goal points:

-   High positive reward for reaching the goal points.
-   Moderate positive reward for following the black line, higher reward when moving straight.
-   Negative reward for deviating from the black line.
-   Negative reward for hitting the boundaries.

### Goal Point Navigation

The car navigates between three goal points (A1, A2, A3) in a loop. The goal points are marked with white circles on the map.

### Painting Tools

The user can draw the black line track using the mouse. The track is stored in a NumPy array and saved as a grayscale image (`sand.jpg`).

### API Buttons

The application provides buttons to clear the track, save the AI's state, and load a previously saved state.

## Usage Instructions

1.  **Draw the Track:** Use the left mouse button to draw the black line track.
2.  **Run the Application:** Execute the `map.py` script.
3.  **Observe the Car's Behavior:** The car will learn to follow the track and navigate between the goal points.
4.  **Save/Load the AI:** Use the save and load buttons to save and load the AI's state.
5.  **Clear the Track:** Use the clear button to clear the track.
