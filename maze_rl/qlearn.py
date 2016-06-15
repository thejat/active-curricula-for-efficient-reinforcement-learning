import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import math
import cv2

class Maze(object):
	def __init__(self, grid_size=9, free_cells = [], goal = []):
		self.grid_size = grid_size
		self.free_cells = free_cells
		self.goal = goal
		self.maze = np.zeros((grid_size,grid_size))
		for i in free_cells:
			self.maze[i[0]][i[1]] = 1
		# print(self.maze)
	def reset(self):
		start_index = np.random.randint(0,len(self.free_cells))
		self.curr_state = free_cells[start_index]

	def state(self):
		return self.curr_state

	def act(self, action):
		if(action == 0):
			next_state = [self.curr_state[0]-1,self.curr_state[1]]
		elif(action == 1):
			next_state = [self.curr_state[0]+1,self.curr_state[1]]
		elif(action == 2):
			next_state = [self.curr_state[0],self.curr_state[1]+1]
		elif(action == 3):
			next_state = [self.curr_state[0],self.curr_state[1]-1]

		if ((next_state in free_cells) or (next_state == self.goal)):
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


if __name__ == "__main__":

	#Parameters of Environment
	grid_size = 11
	# free_cells = [[8,3],[8,4],[8,5],[8,6],[8,7],[7,5],[6,5],[5,5],[4,5]]       # where agent can walk through
	free_cells = [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ] , [ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ] , [ 8,5 ] , [ 9,6 ] , [ 8,6 ] , [ 8,7 ]]
	goal = [8,8]                                       # destination of the agent

	epochs = 200
	epsilon = 1.0
	alpha = 0.9
	discount = 0.9
	num_actions = 4                                    # up, down, right, left

	Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]
	# for i in range(grid_size):
	# 	for j in range(grid_size):
	# 		Q[i][j] = [1,1,1,1]

	env = Maze(grid_size, free_cells, goal)
	c = 0
	for e in range(epochs):
		env.reset()
		game_over = False
		max_iter = 0
		while not (game_over):# or max_iter > 100):
			max_iter += 1
			curr_state = env.state()

			if np.random.rand() <= epsilon/math.sqrt(e+1):
				action = np.random.randint(0, num_actions)
			else:
				action = np.argmax(Q[curr_state[0]][curr_state[1]])

			# print (curr_state, action)

			next_state, reward, game_over = env.act(action)

			#Q-learning update
			Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action]) 

		plot =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
		# plot = np.asarray(plot)
		# im = np.array(plot * 255, dtype = np.uint8)
		# threshed = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
		# img = Image.fromarray(plot)
		# img.show()
		# cv2.imshow('image', plot)
		# cv2.waitkey(10)
		plt.imshow(plot, interpolation='none', cmap='gray')
		# plt.show()
		plt.savefig("images/%03d.png" % c)
		c += 1



