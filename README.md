# MDP Elevator Optimization

A Python-based implementation of a Markov Decision Process (MDP) to control a simplified elevator system using Reinforcement Learning (Value Iteration).

## Project Overview
The goal of this project is to design an optimal control policy for a single elevator in a 3-story building. The elevator must decide whether to move Up, Down, Stay, or Pick Up passengers to minimize total wait time and energy consumption while handling stochastic passenger arrivals.

This project implements:
- A custom **stochastic environment** modeled as an MDP.
- The **Value Iteration algorithm** to mathematically derive the optimal policy.
- A **GUI visualization** (Tkinter) to demonstrate the agent's behavior in real-time.

## The Math (MDP Formulation)
The problem is modeled with the following parameters:

### 1. State Space
The state consists of a tuple `(elevator_position, request_mask)`:
- **Floors:** 1, 2, or 3.
- **Request Mask:** A 3-bit binary number representing pending requests on each floor (e.g., `101` means passengers are waiting on Floors 1 and 3).
- **Total States:** 24 ($3 \text{ floors} \times 8 \text{ mask combinations}$).

### 2. Action Space
- **UP / DOWN:** Moves the elevator one floor (energy cost).
- **STAY:** Helper action to avoid unnecessary movement.
- **PICK:** Clears the request at the current floor (if any).

### 3. Rewards & Penalties
The agent learns based on the following reward structure:
- **+5.0**: Successfully picking up a passenger.
- **-0.5**: Energy cost for every move (Up/Down).
- **-1.0**: Penalty per time step for *each* person currently waiting (minimizes total wait time).

### 4. Stochasticity
- **New Arrivals:** At every time step, there is a **0.1 probability** of a new passenger arriving at any given floor, independent of others.

## How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/BackgroundTree/mdp-elevator.git
   cd mdp-elevator
2. **Run the simulation:**
    ```bash 
    python main.py

## Technologies Used
* **Python 3**: Core logic and scripting.
* **NumPy**: Vectorized calculations for the Value Iteration algorithm.
* **Tkinter**: Standard Python GUI library for the real-time visualization.