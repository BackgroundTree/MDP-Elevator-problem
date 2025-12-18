import tkinter as tk
from elevator import ElevatorEnv, ValueIterationSolver
from run_gui import ElevatorGUI

if __name__ == "__main__":
    print("Initializing Environment...")
    env = ElevatorEnv()

    print("Starting Solver...")
    solver = ValueIterationSolver(env)
    solver.solve()
    print("Solver finished. Optimal policy found!")

    print("Launching Interface...")
    root = tk.Tk()

    app = ElevatorGUI(root, env, solver.policy)
    root.mainloop()