import numpy as np 
import matplotlib.pyplot as plt
import math
# import copy
# import csv
import itertools
import sys
# from random import shuffle
from operator import itemgetter
from maze import Maze

if __name__ == "__main__":

	Subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ]], [[ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ]], [[ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]

	target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

	no_tasks = len(Subtasks)+1
	grid_size = 7
	print ('total no. of tasks: ',no_tasks)
	steps_per_tasks = 2000
	# shuffle(subtasks)

	perm = 0

	for T in itertools.permutations(Subtasks): 
		perm += 1
		Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]
		round_no = 0
		tot_steps = 0
		subtasks = list(T)
		while (len(subtasks)):
			print (round_no)
			round_no += 1
			reward_list = []
			task_no = 0
			
			for task in subtasks:
				# print task
				task_no += 1
				step = 0
				# Parameters of Environment
				free_cells = task[:-1]
				goal = task[-1]
				# task += 1                                 # destination of the agent
				epsilon = 0.3
				alpha = 0.6
				discount = 0.9
				num_actions = 4                                    # up, down, right, left
				env = Maze(grid_size, free_cells, goal)
				env.draw("curr_1/tasks", perm, round_no, task_no)
				exceed = 0
				tot_reward = 0
				while (step < steps_per_tasks):
					env.reset()
					game_over = False
					max_iter = 50
					itr = 0
					while not (game_over or itr > max_iter):
						itr += 1
						curr_state = env.state()
						if np.random.rand() <= epsilon:
							action = np.random.randint(0, num_actions)
						else:
							action = np.argmax(Q[curr_state[0]][curr_state[1]])
						next_state, reward, game_over = env.act(action)
						tot_reward += reward
						step += 1
						# Q-learning update
						Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])

					if(itr > max_iter):
						exceed += 1
						# print ('maximum steps exceeded, starting new episode.')

				print('Exceed: %d' % exceed)
				reward_list.append([task, tot_reward])
				tot_steps += step
			reward_list = sorted(reward_list, key = itemgetter(1))

			if(len(reward_list) == 1):
				num = 0
			else:
				num = len(reward_list)/2 + len(reward_list)%2

			subtasks = []
			print (len(reward_list))
			print (num)
			print (reward_list)
			for i in range(num-1,-1,-1):
				subtasks.append(reward_list[i][0])

		# Learning of Target Task
		free_cells = target_task[:-1]
		goal = target_task[-1]

		epsilon = 0.3
		alpha = 0.6
		discount = 0.9
		num_actions = 4                                    # up, down, right, left

		env = Maze(grid_size, free_cells, goal)
		env.draw("curr_1/tasks", perm, round_no, 6)

		tot_reward = 0
		r_list = [] # reward list for target task
		s_list = [] # step list for target task
		step = 0
		exceed = 0
		while(step < steps_per_tasks):
			env.reset()
			game_over = False
			max_iter = 100
			itr = 0
			while not (game_over or itr > max_iter):
				itr += 1
				curr_state = env.state()

				if np.random.rand() <= epsilon:
					action = np.random.randint(0, num_actions)
				else:
					action = np.argmax(Q[curr_state[0]][curr_state[1]])

				next_state, reward, game_over = env.act(action)
				tot_reward += reward
				step += 1
				r_list.append(tot_reward)
				s_list.append(step+tot_steps)
				# Q-learning update
				Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])
			if(itr > max_iter):
				exceed += 1
				# print ('maximum steps exceeded, starting new episode.')

		print('Exceed: %d' % exceed)
		tot_steps += step

		plt.figure(0)
		plot =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
		plt.imshow(plot, interpolation='none', cmap='gray')
		plt.savefig("curr_1/policies/curr_policy%d.png" % perm)

		plt.figure(1)
		plt.plot(s_list, r_list)
		plt.xscale('log')
		# plt.yscale('log')
		plt.savefig('curr_1/result_%d_1.png' % perm)

		print (r_list[0], s_list[0])
		plt.figure(2)
		plt.plot(s_list,r_list)
		# plt.xscale('log')
		plt.savefig('curr_1/result_%d_2.png' % perm)

	# BASELINE

	tot_reward = 0
	r_list = [] # reward list for target task
	s_list = [] # step list for target task
	step = 0

	epsilon = 0.3
	alpha = 0.6
	discount = 0.9
	num_actions = 4 

	Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]
	print (tot_steps)
	exceed = 0
	while(step < tot_steps):
		env.reset()
		game_over = False
		max_iter = 100
		itr = 0
		while not (game_over or itr > max_iter):
			itr += 1
			curr_state = env.state()

			if np.random.rand() <= epsilon:
				action = np.random.randint(0, num_actions)
			else:
				action = np.argmax(Q[curr_state[0]][curr_state[1]])

			next_state, reward, game_over = env.act(action)
			tot_reward += reward
			step += 1
			r_list.append(tot_reward)
			s_list.append(step)
			# sys.stdout.write('\rstep %i, tot_reward %i, %i' % (step,tot_reward,step-tot_reward))
			# sys.stdout.flush()

			# Q-learning update
			Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])

		if(itr > max_iter):
			exceed += 1
			# print ('maximum steps exceeded, starting new episode.')

	print('Exceed: %d' % exceed)
	plt.figure(0)
	plt.clf()
	plot =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
	plt.imshow(plot, interpolation='none', cmap='gray')
	plt.savefig("curr_1/base_policy.png")

	plt.figure(1)
	plt.plot(s_list, r_list)
	# plt.xscale('log')
	plt.yscale('log')
	plt.savefig('curr_1/result_1.png')

	plt.figure(2)
	plt.plot(s_list, r_list)
	# plt.xscale('log')
	plt.savefig('curr_1/result_2.png')

	# print(r_list[0], s_list[0])
