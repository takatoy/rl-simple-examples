import os
import time
import tkinter as tk
from PIL import Image, ImageTk


class Env:
    action_space = None

    def step(self, action):
        """
        Takes an action, executes, and returns tuple (observation, reward, done).
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the environment, returns initial observation.
        """
        raise NotImplementedError

    def render(self):
        """
        Render the environment.
        """
        raise NotImplementedError


class GridWorld(Env):
    action_space = [0, 1, 2, 3]  # [up, down, right, left]
    move_map = [(0, -1), (0, 1), (1, 0), (-1, 0)]

    def __init__(self):
        self.col = 6
        self.row = 5
        self.stage = [  # 0: empty, 1: wall, 2: trap, 3: start, 4: goal
            [3, 0, 0, 0, 2, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 2, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 2, 0, 1, 4]
        ]
        self.max_steps = 100

        self.state = [0, 0]
        self.n_steps = 0

        # for rendering
        self.window = None
        self.canvas = None
        self.width = 620
        self.height = 520
        self.margin = 10
        self.cell_width = (self.width - self.margin * 2) // self.col
        self.cell_height = (self.height - self.margin * 2) // self.row

        self.agent_img = Image.open('assets/agent.jpg').resize(
            (self.cell_width - 30, self.cell_height - 30), Image.ANTIALIAS)
        self.wall_img = Image.open('assets/wall.jpg').resize(
            (self.cell_width - 30, self.cell_height - 30), Image.ANTIALIAS)
        self.trap_img = Image.open('assets/trap.jpg').resize(
            (self.cell_width - 30, self.cell_height - 30), Image.ANTIALIAS)
        self.goal_img = Image.open('assets/goal.jpg').resize(
            (self.cell_width - 30, self.cell_height - 30), Image.ANTIALIAS)
        self.tk_wall_img = []
        self.tk_trap_img = []
        self.tk_goal_img = []

    def step(self, action):
        assert(action in self.action_space)

        self.n_steps += 1

        nx = self.state[0] + self.move_map[action][0]
        ny = self.state[1] + self.move_map[action][1]

        done = False
        if (ny >= 0 and ny < self.row and
            nx >= 0 and nx < self.col and
            self.stage[ny][nx] != 1):
            if (self.stage[ny][nx] == 2 or
                self.stage[ny][nx]):
                done = True
            self.state = [nx, ny]

        reward = -0.1
        if self.stage[self.state[1]][self.state[0]] == 4:
            # goal reward
            reward = 5.0
        elif self.stage[self.state[1]][self.state[0]] == 2:
            # trap reward
            reward = -1.0

        if self.n_steps >= self.max_steps:
            reward = -5.0
            done = True

        return list(self.state), reward, done

    def reset(self):
        self.n_steps = 0
        self.state = [0, 0]
        return list(self.state)

    def render(self, wait=0.05):
        if self.window is None:
            self.window = tk.Tk()
            self.window.title('Demo')
            self.canvas = tk.Canvas(self.window, width=self.width, height=self.height, bg="white")
            self.canvas.pack()

            # create board
            x = self.margin
            for i in range(self.col + 1):
                self.canvas.create_line(x, self.margin, x, self.height - self.margin)
                x += self.cell_width

            y = self.margin
            for i in range(self.row + 1):
                self.canvas.create_line(self.margin, y, self.width - self.margin, y)
                y += self.cell_height

            for y in range(self.row):
                for x in range(self.col):
                    if self.stage[y][x] == 1:
                        im = ImageTk.PhotoImage(self.wall_img)
                        self.tk_wall_img.append(im)
                        self.canvas.create_image(
                            self.margin + self.cell_width * x + self.cell_width // 2,
                            self.margin + self.cell_height * y + self.cell_height // 2,
                            image=im)
                    elif self.stage[y][x] == 2:
                        im = ImageTk.PhotoImage(self.trap_img)
                        self.tk_trap_img.append(im)
                        self.canvas.create_image(
                            self.margin + self.cell_width * x + self.cell_width // 2,
                            self.margin + self.cell_height * y + self.cell_height // 2,
                            image=im)
                    elif self.stage[y][x] == 4:
                        im = ImageTk.PhotoImage(self.goal_img)
                        self.tk_goal_img.append(im)
                        self.canvas.create_image(
                            self.margin + self.cell_width * x + self.cell_width // 2,
                            self.margin + self.cell_height * y + self.cell_height // 2,
                            image=im)
        else:
            self.canvas.delete('agent')

        im = ImageTk.PhotoImage(self.agent_img)
        self.canvas.create_image(
            self.margin + self.cell_width * self.state[0] + self.cell_width // 2,
            self.margin + self.cell_height * self.state[1] + self.cell_height // 2,
            image=im,
            tag='agent')

        self.window.update()
        if wait > 0: time.sleep(wait)  # speed

    def getProbability(self, state, action):
        pass
