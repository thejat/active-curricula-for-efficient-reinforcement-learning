import numpy as np 
import matplotlib.pyplot as plt
import math
# import cv2
import copy
import csv
import itertools

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

	def draw(self, perm, task):
		grid = np.zeros((self.grid_size, self.grid_size))
		for i in self.free_cells:
			grid[i[1]][i[0]] = 0.5
		grid[self.goal[1]][self.goal[0]] = 1
		plt.imshow(grid, interpolation='none', cmap='gray')
		plt.savefig("perm/%d/subtask/%02d.png" % (perm,task))

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

	subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ]], [[ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]

	target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

	no_tasks = len(subtasks)+1
	grid_size = 7
	print ('total no. of tasks: ',no_tasks)
	perm = 0              # to track no. of permutations
	perm_episodes = []
	for T in itertools.permutations(subtasks):
		perm += 1
		print ('permutation no.: ',perm)
		tasks = list(T)
		tasks.append(target_task)             # final task is target task

		tot_episodes = []      # to count how many episodes required for each task for a particular permutation

		Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]

		task = 0
		while(task < no_tasks):
			#Parameters of Environment

			free_cells = tasks[task][:-1]
			goal = tasks[task][-1]    
			task += 1                                 # destination of the agent
			# print (free_cells, goal)

			# epochs = 20
			epsilon = 0.5
			alpha = 0.8
			discount = 0.9
			num_actions = 4                                    # up, down, right, left

			# Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]
			# for i in range(grid_size):
			# 	for j in range(grid_size):
			# 		Q[i][j] = [1,1,1,1]

			env = Maze(grid_size, free_cells, goal)
			env.draw(perm, task)
			task_epi = 0   # no. of episodes per task
			stop_task = 0  # to check whether task is completed or not
			thres = 0.0
			episode = 0
			# episode = 15*task          # to compensate for previously learned tasks, we don't want much exploration previous knowledge has been transferred

			while not stop_task:
			# for e in range(2):
				episode += 1
				env.reset()
				game_over = False
				max_iter = 0
				Q_prev = copy.deepcopy(Q)
				while not (game_over):# or max_iter > 300):
					max_iter += 1
					curr_state = env.state()

					if np.random.rand() <= epsilon/episode:
						action = np.random.randint(0, num_actions)
					else:
						action = np.argmax(Q[curr_state[0]][curr_state[1]])

					# print (curr_state, action)

					next_state, reward, game_over = env.act(action)

					#Q-learning update
					Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action]) 

				if True:#(max_iter != 300):
					for i in free_cells:
						prev_val = sum(Q_prev[i[0]][i[1]])
						new_val = sum(Q[i[0]][i[1]])
						# print (prev_val, new_val)
						if(abs(prev_val - new_val) > thres):
							stop_task = 0
							break
						else:
							stop_task = 1

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
				plt.savefig("perm/%d/%02d_%03d.png" % (perm,task,task_epi))
				task_epi += 1

			tot_episodes.append(task_epi)

		print (tot_episodes)
		perm_episodes.append(tot_episodes)

	with open("output.csv", "wb") as f:
		writer = csv.writer(f)
		writer.writerows(perm_episodes)
