import numpy as np
import matplotlib.pyplot as plt

class Maze(object):
	def __init__(self, grid_size, free_cells = [], goal = []):
		self.grid_size = grid_size
		self.num_actions = 4
		self.free_cells = free_cells
		self.goal = goal
		self.maze = np.zeros((grid_size,grid_size))
		for i in free_cells:
			self.maze[i[0]][i[1]] = 1

	def reset(self):
		start_index = np.random.randint(0,len(self.free_cells))
		self.curr_state = self.free_cells[start_index]

	def state(self):
		return self.curr_state

	def draw(self, task_no):
		grid = np.zeros((self.grid_size, self.grid_size))
		for i in self.free_cells:
			grid[i[1]][i[0]] = 0.5
		grid[self.goal[1]][self.goal[0]] = 1
		plt.figure(0)
		plt.clf()
		plt.imshow(grid, interpolation='none', cmap='gray')
		plt.savefig("inter/tasks/%d.png" % task_no)

	def act(self, action):
		if(action == 0):
			next_state = [self.curr_state[0]-1,self.curr_state[1]]
		elif(action == 1):
			next_state = [self.curr_state[0]+1,self.curr_state[1]]
		elif(action == 2):
			next_state = [self.curr_state[0],self.curr_state[1]+1]
		elif(action == 3):
			next_state = [self.curr_state[0],self.curr_state[1]-1]

		if ((next_state in self.free_cells) or (next_state == self.goal)):
			self.curr_state = next_state
		else:
			next_state = self.curr_state

		if(next_state == self.goal):
			reward = 1
			game_over = True
		else:
			reward = 0
			game_over = False

		return next_state, reward, game_over