import numpy as np
import matplotlib.pyplot as plt

class Grid(object):
	def __init__(self, grid_size, block_cells = [], goal = [], start = []):
		self.grid_size = grid_size
		self.num_actions = 4
		self.block_cells = block_cells
		self.goal = goal
		self.start = start
		self.grid = np.zeros((grid_size,grid_size))
		for i in self.block_cells:
			self.grid[i[0]][i[1]] = 1

	def reset(self):
		self.curr_state = self.start
		# print(self.curr_state)

	def state(self):
		return self.curr_state

	def draw(self, path = "", num1 = 1, num2 = 1, num3 = 1):
		self.grid = np.zeros((self.grid_size, self.grid_size))
		self.grid[:] = 1
		for i in self.block_cells:
			self.grid[i[1]][i[0]] = 0
		self.grid[self.goal[1]][self.goal[0]] = 0.25
		self.grid[self.start[1]][self.start[0]] = 0.7
		plt.figure(0)
		plt.clf()
		fig = plt.imshow(self.grid, interpolation='nearest',cmap='hot')
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

		if ((self.next_state in self.block_cells) or (self.next_state[0] not in np.arange(self.grid_size)) or (self.next_state[1] not in np.arange(self.grid_size))):
			self.next_state = self.curr_state
			self.reward = 0
		else:
			self.curr_state = self.next_state
			self.reward = 0

		if(self.curr_state == self.goal):
			self.reward = 1
			# printf('yes')
			self.game_over = True
		else:
			self.game_over = False

		return self.next_state, self.reward, self.game_over