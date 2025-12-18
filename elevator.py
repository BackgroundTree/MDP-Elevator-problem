import numpy as np
import random

NUM_FLOORS = 3
NUM_REQUEST_OPTS = 2 ** NUM_FLOORS  #2^3 = 8 possibilities (000 to 111)
TOTAL_STATES = NUM_FLOORS * NUM_REQUEST_OPTS  #24 total states

#Actions mapped to integers
ACT_UP = 0
ACT_DOWN = 1
ACT_STAY = 2
ACT_PICK = 3
ACTIONS = [ACT_UP, ACT_DOWN, ACT_STAY, ACT_PICK]
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "STAY", 3: "PICK"}


class ElevatorEnv:
    def __init__(self):
        self.num_floors = NUM_FLOORS
        self.state_space_size = TOTAL_STATES

        #probs n rewards
        self.prob_new_req = 0.1
        self.reward_pick = 5.0
        self.reward_move = -0.5
        self.reward_wait_per_req = -1.0

    def encode_state(self, floor, mask):
        """
        Converts (floor, mask) to a unique state ID (0-23)
        Floor input is expected to be 1-based (1, 2, 3)
        """
        #Convert 1-based floor to 0-based index
        floor_idx = floor - 1

        #Formula: floor_index * 8 + mask
        #Example: Floor 1 (idx 0), Mask 5 -> 0 * 8 + 5 = 5
        #Example: Floor 2 (idx 1), Mask 5 -> 1 * 8 + 5 = 13
        return (floor_idx * NUM_REQUEST_OPTS) + mask

    def decode_state(self, state_id):
        """
        Converts unique state ID back to (floor, mask)
        Returns floor as 1-based (1, 2, 3)
        """
        mask = state_id % NUM_REQUEST_OPTS
        floor_idx = state_id // NUM_REQUEST_OPTS
        return floor_idx + 1, mask

    def print_state_details(self, state_id):
        """Helper to visualize what a state ID actually means"""
        floor, mask = self.decode_state(state_id)
        #Convert mask to binary string, e.g., 5 -> '101'
        #zfill(3) ensures it shows as 001 instead of just 1
        binary_mask = format(mask, 'b').zfill(3)
        print(f"ID {state_id:02d} => Floor: {floor} | Requests: {binary_mask} (F3, F2, F1)")

    def get_transitions(self, state_id, action):
        """
        Returns a list of (probability, next_state, reward) for the given state and action.
        This represents the 'Model' of the environment needed for the solver.
        """
        curr_floor, curr_mask = self.decode_state(state_id)

        #Phase 1: Deterministic action effects
        next_floor = curr_floor
        temp_mask = curr_mask
        reward = 0.0

        #1. Calculate WAIT Penalty (based on current status)
        #Count bits set to 1 in the mask
        num_waiting = bin(curr_mask).count('1')
        reward += (num_waiting * self.reward_wait_per_req)

        #2.Apply Action Logic
        if action == ACT_UP:
            if curr_floor < self.num_floors:
                next_floor += 1
                reward += self.reward_move
            #If at top, stay (no move penalty)

        elif action == ACT_DOWN:
            if curr_floor > 1:
                next_floor -= 1
                reward += self.reward_move

        elif action == ACT_PICK:
            #Check for request at current floor
            #create bitmask for current floor (e.g., Floor 1->1, Floor 2->2, Floor 3->4)
            floor_bit = 1 << (curr_floor - 1)

            if (curr_mask & floor_bit):  #If bit is set
                reward += self.reward_pick
                temp_mask = curr_mask & ~floor_bit  #Clear bit
            else:
                #Optional: Penalty for false pick? PDF says "Optional -5".
                #Let's leave it 0 or small penalty to prevent spamming.
                pass

                # (ACT_STAY does nothing to floor or mask, just incurs wait penalty)

        #PHASE 2: Stochastic new arrivals
        #must iterate through all 8 possibilities of new arrivals (000 to 111)
        #to calculate the Expected Value later.

        transitions = []

        #Iterate 0 to 7 (all possible arrival patterns)
        for arrival_mask in range(NUM_REQUEST_OPTS):
            prob = 1.0

            #Calculate probability of this specific arrival_mask
            #For each floor, did a person appear?
            for f in range(self.num_floors):
                #Check bit f in arrival_mask
                is_arrival = (arrival_mask >> f) & 1

                if is_arrival:
                    prob *= self.prob_new_req  #0.1
                else:
                    prob *= (1.0 - self.prob_new_req)  #0.9

            #The final mask is the Union (OR) of remaining requests + new arrivals
            final_mask = temp_mask | arrival_mask

            #Encode back to state ID
            next_state_id = self.encode_state(next_floor, final_mask)

            transitions.append((prob, next_state_id, reward))

        return transitions

    def step(self, state_id, action):
        curr_floor, curr_mask = self.decode_state(state_id)

        #Initialize step reward
        reward = 0.0

        #Penalty for Waiting Passengers
        num_waiting = bin(curr_mask).count('1')
        reward += (num_waiting * self.reward_wait_per_req)

        #Deterministic Action (Move/Pick)
        next_floor = curr_floor
        temp_mask = curr_mask

        if action == ACT_UP:
            if curr_floor < self.num_floors:
                next_floor += 1
                reward += self.reward_move

        elif action == ACT_DOWN:
            if curr_floor > 1:
                next_floor -= 1
                reward += self.reward_move

        elif action == ACT_PICK:
            floor_bit = 1 << (curr_floor - 1)
            if (curr_mask & floor_bit):
                temp_mask = curr_mask & ~floor_bit
                reward += self.reward_pick

        #Stochastic New Arrivals
        arrival_mask = 0
        for f in range(self.num_floors):
            if random.random() < self.prob_new_req:
                arrival_mask |= (1 << f)

        final_mask = temp_mask | arrival_mask

        return self.encode_state(next_floor, final_mask), reward


class ValueIterationSolver:
    def __init__(self, env, gamma=0.9, theta=1e-4):
        self.env = env
        self.gamma = gamma
        self.theta = theta  # Convergence threshold (when to stop)
        #Initialize Value table with zeros
        self.V = np.zeros(env.state_space_size)
        #Initialize Policy table (will store best action ID for each state)
        self.policy = np.zeros(env.state_space_size, dtype=int)

    def solve(self):
        """
        Runs the Value Iteration algorithm.
        """
        iteration = 0
        print("Starting Value Iteration...")

        while True:
            delta = 0
            #Create a copy to store new values
            new_V = np.copy(self.V)

            for state in range(self.env.state_space_size):
                q_values = []

                #Try all 4 actions
                for action in ACTIONS:
                    q_val = 0
                    transitions = self.env.get_transitions(state, action)

                    #Bellman Equation: Sum [ Prob * (Reward + Gamma * V_old(next_state)) ]
                    for prob, next_state, reward in transitions:
                        q_val += prob * (reward + self.gamma * self.V[next_state])

                    q_values.append(q_val)

                #Find the best action value
                best_val = max(q_values)

                #Calculate the difference (delta) for convergence check
                delta = max(delta, abs(best_val - self.V[state]))

                #Update the new table
                new_V[state] = best_val

            self.V = new_V
            iteration += 1

            #Check convergence
            if delta < self.theta:
                print(f"Converged in {iteration} iterations!")
                break

        #After convergence, extract the best policy one last time
        self._extract_policy()

    def _extract_policy(self):
        """
        Derives the optimal policy from the final Value table.
        """
        for state in range(self.env.state_space_size):
            q_values = []
            for action in ACTIONS:
                q_val = 0
                transitions = self.env.get_transitions(state, action)
                for prob, next_state, reward in transitions:
                    q_val += prob * (reward + self.gamma * self.V[next_state])
                q_values.append(q_val)

            #Store the index of the action with the highest Q-value
            self.policy[state] = np.argmax(q_values)


#TEST BLOCK
if __name__ == "__main__":
    env = ElevatorEnv()
    print("Testing State Encoding:")
    #Test State: Floor 2, Requests on F1 and F3 (Mask 101 = 5)
    s_id = env.encode_state(2, 5)
    env.print_state_details(s_id)

    print("\n-------------Testing Transitions------------")
    #Scenario: Elevator at Floor 2 (ID 1), Request at Floor 2 (Mask 2 -> '010')
    #Combined State ID: 1 * 8 + 2 = 10
    start_state = env.encode_state(2, 2)

    print(f"Start State: {start_state} (Floor 2, Request F2)")
    print("Action: PICK")

    #Get transitions
    outcomes = env.get_transitions(start_state, ACT_PICK)

    #Check the first outcome (Case where NO new people appear)
    #The arrival mask for "no new people" is 0.
    #Probability should be 0.9 * 0.9 * 0.9 = 0.729

    prob, next_s, rew = outcomes[0]
    next_f, next_m = env.decode_state(next_s)

    print(f"Outcome 'No New Arrivals' (Prob={prob:.3f}):")
    print(f"  -> Next Floor: {next_f}")
    print(f"  -> Next Mask: {next_m}")
    print(f"  -> Reward: {rew}")

    print("\n---------------Running Solver---------------")
    solver = ValueIterationSolver(env)
    solver.solve()

    print("\n--------Optimal Policy Demonstration--------")

    #Test ase 1: Floor 2, Request on Floor 2
    #Should PICK
    s_id = env.encode_state(2, 2)  #F2, Mask 010
    best_act = solver.policy[s_id]
    print(f"State: Floor 2, Req F2 -> Best Action: {ACTION_NAMES[best_act]}")

    #Test case 2: Floor 1, Request on Floor 3
    #Should Move UP
    s_id = env.encode_state(1, 4)  # F1, Mask 100
    best_act = solver.policy[s_id]
    print(f"State: Floor 1, Req F3 -> Best Action: {ACTION_NAMES[best_act]}")

    #Test case 3: Floor 3, Request on Floor 1
    #Should Move DOWN
    s_id = env.encode_state(3, 1)  #F3, Mask 001
    best_act = solver.policy[s_id]
    print(f"State: Floor 3, Req F1 -> Best Action: {ACTION_NAMES[best_act]}")