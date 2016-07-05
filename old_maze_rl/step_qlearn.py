import numpy as np 
import matplotlib.pyplot as plt
import math
# import copy
# import csv
import itertools
import sys

class Maze(object):
	def __init__(self, grid_size=9, free_cells = [], goal = []):
		self.grid_size = grid_size
		self.free_cells = free_cells
		self.goal = goal
		self.maze = np.zeros((self.grid_size, self.grid_size))
		for i in self.free_cells:
			self.maze[i[0]][i[1]] = 1

	def reset(self):
		self.start_index = np.random.randint(0,len(self.free_cells))
		self.curr_state = self.free_cells[self.start_index]

	def state(self):
		return self.curr_state

	def draw(self, perm, task):
		self.grid = np.zeros((self.grid_size, self.grid_size))
		for i in self.free_cells:
			self.grid[i[1]][i[0]] = 0.5
		self.grid[self.goal[1]][self.goal[0]] = 1
		plt.imshow(self.grid, interpolation='none', cmap='gray')
		plt.savefig("perm2/%d/%02d.png" % (perm,task))

	def act(self, action):
		if(action == 0):
			self.next_state = [self.curr_state[0]-1,self.curr_state[1]]
		elif(action == 1):
			self.next_state = [self.curr_state[0]+1,self.curr_state[1]]
		elif(action == 2):
			self.next_state = [self.curr_state[0],self.curr_state[1]+1]
		elif(action == 3):
			self.next_state = [self.curr_state[0],self.curr_state[1]-1]

		if ((self.next_state in self.free_cells) or (self.next_state == self.goal)):
			self.curr_state = self.next_state
		else:
			self.next_state = self.curr_state

		if(self.next_state == self.goal):
			self.reward = 1
			self.game_over = True
		else:
			self.reward = 0
			self.game_over = False
		return self.next_state, self.reward, self.game_over

if __name__ == "__main__":

	subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ]], [[ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]

	target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

	no_tasks = len(subtasks)+1
	grid_size = 7
	print ('total no. of tasks: ',no_tasks)
	perm = 0              # to track no. of permutations
	# perm_episodes = []
	tot_steps = 2500000
	for T in itertools.permutations(subtasks):
		step = 0
		perm += 1
		# if (perm < 16):
		# 	continue
		print ('permutation no.: ',perm)
		tasks = list(T)
		tasks.append(target_task)             # final task is target task

		# tot_episodes = []      # to count how many episodes required for each task for a particular permutation

		Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]
		Q_prev = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]

		perm_reward = []
		perm_steps = []
		tot_reward = 0

		task = 0
		while(task < no_tasks):
			#Parameters of Environment

			free_cells = tasks[task][:-1]
			goal = tasks[task][-1]
			task += 1                                 # destination of the agent

			epsilon = 0.7
			alpha = 0.6
			discount = 0.9
			num_actions = 4                                    # up, down, right, left

			env = Maze(grid_size, free_cells, goal)
			env.draw(perm, task)
			task_epi = 0   # no. of episodes per task
			stop_task = 0  # to check whether task is completed or not
			thres = 0.0
			# episode = 15*task          # to compensate for previously learned tasks, we don't want much exploration previous knowledge has been transferred

			# print (perm, task)
			while ((not stop_task) and task != no_tasks) or (task == no_tasks and step < tot_steps):
				# sys.stdout.write('\rstep %i' % step)
				# sys.stdout.flush()
				env.reset()
				game_over = False
				max_iter = 0
				for cell in free_cells:
					for act in range(num_actions):
						Q_prev[cell[0]][cell[1]][act] = Q[cell[0]][cell[1]][act]

				while not (game_over or max_iter > 20000):
					max_iter += 1
					curr_state = env.state()

					if np.random.rand() <= epsilon/(task_epi+1):
						action = np.random.randint(0, num_actions)
					else:
						action = np.argmax(Q[curr_state[0]][curr_state[1]])

					# print (curr_state, action)

					next_state, reward, game_over = env.act(action)
					tot_reward += reward
					step += 1
					# sys.stdout.write('\rstep %i, tot_reward %i, %i' % (step,tot_reward,step-tot_reward))
					# sys.stdout.flush()
					if(task == no_tasks):
						perm_reward.append(tot_reward)
						perm_steps.append(step)

					#Q-learning update
					Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])
				# print (max_iter)
				if (max_iter < 20000 and task != no_tasks):
					for i in free_cells:
						prev_val = sum(Q_prev[i[0]][i[1]])
						new_val = sum(Q[i[0]][i[1]])
						# print (prev_val, new_val)
						if(abs(prev_val - new_val) > thres):
							stop_task = 0
							break
						else:
							stop_task = 1
				# elif(max_iter >= 20000):
					# print ('iterations exceeded, starting new episode')

				# plot =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
				# plt.imshow(plot, interpolation='none', cmap='gray')
				# plt.savefig("perm/%d/%02d_%03d.png" % (perm,task,task_epi))
				task_epi += 1

			# tot_episodes.append(task_epi)

		# plt.figure(0)
		# plt.clf()
		print (perm, Q)
		plt.figure(perm)
		plot =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
		plt.imshow(plot, interpolation='none', cmap='gray')
		plt.savefig("%d_policy.png" % (perm))


		print('plotting')
		plt.figure(1)
		plt.plot(perm_steps,perm_reward)
		# plt.gca().set_ylim(bottom=0)
		plt.xscale('log')
		plt.yscale('log')
		# plt.gca().invert_yaxis()
		plt.savefig('resutl%d_1.png' % perm)
		print ('done')
		print('plotting')

		plt.figure(2)
		plt.plot(perm_steps,perm_reward)
		# plt.gca().set_ylim(bottom=0)
		plt.xscale('log')
		# plt.yscale('log')
		# plt.gca().invert_yaxis()
		plt.savefig('resutl%d_2.png' % perm)
		print ('done')

		# print (tot_episodes)
		# perm_episodes.append(tot_episodes)
		print (step)

	plt.figure(1)
	plt.savefig('result_1.png')
	plt.figure(2)
	plt.savefig('result_2.png')
	with open("output.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(perm_episodes)
