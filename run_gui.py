import tkinter as tk
from elevator import ACTION_NAMES

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
DELAY_MS = 800  #Speed of simulation

# Colors / Theme
BG_COLOR = "#F0F0F0"
SHAFT_COLOR = "#FFFFFF"
FLOOR_LINE_COLOR = "#333333"
ELEV_FILL = "#C0C0C0"
ELEV_OUTLINE = "#666666"
PERSON_COLOR = "#D9534F"
TEXT_COLOR = "#000000"


class ElevatorGUI:
    def __init__(self, master, env, policy):
        self.master = master
        self.env = env
        self.policy = policy

        #State & Score Tracking
        self.current_state = env.encode_state(1, 0)
        self.total_reward = 0.0  #<- New Score Counter

        self.master.title("MDP Elevator Optimization Project")
        self.master.configure(bg=BG_COLOR)

        self.canvas = tk.Canvas(master, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg=BG_COLOR, highlightthickness=0)
        self.canvas.pack(pady=10)

        # Status Label
        self.status_var = tk.StringVar()
        self.status_var.set("Starting simulation...")
        self.label = tk.Label(master, textvariable=self.status_var, font=("Helvetica", 12, "bold"), bg=BG_COLOR,
                              fg=TEXT_COLOR)
        self.label.pack(pady=(0, 10))

        self.floor_height = WINDOW_HEIGHT // 3
        self.shaft_left = 100
        self.shaft_right = 250

        self.update_gui()

    def draw_building(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(self.shaft_left, 0, self.shaft_right, WINDOW_HEIGHT, fill=SHAFT_COLOR, outline="")

        for f_idx in range(3):
            y = f_idx * self.floor_height
            floor_num = 3 - f_idx
            self.canvas.create_line(20, y + self.floor_height, WINDOW_WIDTH - 20, y + self.floor_height,
                                    fill=FLOOR_LINE_COLOR, width=3)
            self.canvas.create_text(50, y + self.floor_height - 30, text=f"Floor {floor_num}",
                                    font=("Helvetica", 16, "bold"), fill=TEXT_COLOR)

    def draw_elevator_and_people(self):
        curr_floor, mask = self.env.decode_state(self.current_state)

        #Draw Passengers
        person_x_start = self.shaft_right + 50
        for f_idx in range(3):
            floor_num = f_idx + 1
            if (mask >> f_idx) & 1:
                visual_floor_idx = 3 - floor_num
                base_y = visual_floor_idx * self.floor_height + self.floor_height
                head_y = base_y - 60

                self.canvas.create_oval(person_x_start, head_y, person_x_start + 20, head_y + 20, fill=PERSON_COLOR,
                                        outline=PERSON_COLOR)
                self.canvas.create_line(person_x_start + 10, head_y + 20, person_x_start + 10, base_y - 20,
                                        fill=PERSON_COLOR, width=3)
                self.canvas.create_text(person_x_start + 40, base_y - 40, text="WAITING",
                                        font=("Helvetica", 12, "bold"), fill=PERSON_COLOR, anchor="w")

        #Draw Elevator
        elev_top_y = (3 - curr_floor) * self.floor_height + 10
        elev_bottom_y = elev_top_y + self.floor_height - 20
        self.canvas.create_rectangle(self.shaft_left + 10, elev_top_y, self.shaft_right - 10, elev_bottom_y,
                                     fill=ELEV_FILL, outline=ELEV_OUTLINE, width=4)
        mid_x = (self.shaft_left + self.shaft_right) // 2
        self.canvas.create_line(mid_x, elev_top_y + 5, mid_x, elev_bottom_y - 5, fill=ELEV_OUTLINE, width=2)

    def update_gui(self):
        #Get Best Action
        action = self.policy[self.current_state]
        action_name = ACTION_NAMES[action]

        #Execute Step
        next_state, step_reward = self.env.step(self.current_state, action)

        #Update Totals
        self.total_reward += step_reward
        self.current_state = next_state  # Update state for next loop

        #Update Status Text (Points instead of Floor)
        status_text = f"Action: {action_name} | Step Reward: {step_reward:.1f} | Total Points: {self.total_reward:.1f}"
        self.status_var.set(status_text)

        #Redraw
        self.draw_building()
        self.draw_elevator_and_people()
        self.master.after(DELAY_MS, self.update_gui)