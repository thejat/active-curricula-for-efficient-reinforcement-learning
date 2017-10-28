import numpy as np
import matplotlib.pyplot as plt

class Maze(object):
	def __init__(self, grid_size, free_cells = [], goal = []):
		self.grid_size = grid_size
		self.num_actions = 4
		self.free_cells = free_cells
		self.goal = goal
		self.maze = np.zeros((grid_size,grid_size))
		for i in self.free_cells:
			self.maze[i[0]][i[1]] = 1

	def reset(self):
		self.start_index = np.random.randint(0,len(self.free_cells))
		self.curr_state = self.free_cells[self.start_index]
		# print(self.curr_state)

	def state(self):
		return self.curr_state

	def draw(self, path = "", num1 = 1, num2 = 1, num3 = 1):
		self.grid = np.zeros((self.grid_size, self.grid_size))
		for i in self.free_cells:
			self.grid[i[1]][i[0]] = 1
		self.grid[self.goal[1]][self.goal[0]] = 0.5
		plt.figure(0)
		plt.clf()
		fig = plt.imshow(self.grid, interpolation='nearest',cmap='hot')
		fig = plt.imshow(self.grid, interpolation='nearest',cmap='nipy_spectral')

		# plt.axis('off')
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)
		plt.savefig(path + "%d_%d_%d.png" % (num1, num2, num3), bbox_inches='tight', pad_inches = 0)

	def act(self, action):
		if(action == -1):
			self.next_state = self.curr_state
		elif(action == 0):
			self.next_state = [self.curr_state[0]-1,self.curr_state[1]]
		elif(action == 1):
			self.next_state = [self.curr_state[0]+1,self.curr_state[1]]
		elif(action == 2):
			self.next_state = [self.curr_state[0],self.curr_state[1]+1]
		elif(action == 3):
			self.next_state = [self.curr_state[0],self.curr_state[1]-1]

		if ((self.next_state in self.free_cells) or (self.next_state == self.goal)):
			self.curr_state = self.next_state
			self.reward = 0
		else:
			self.next_state = self.curr_state
			self.reward = 0

		if(self.next_state == self.goal):
			self.reward = 1
			# printf('yes')
			self.game_over = True
		else:
			self.game_over = False

		return self.next_state, self.reward, self.game_over