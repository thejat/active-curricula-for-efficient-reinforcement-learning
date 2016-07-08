import numpy as np 
import matplotlib.pyplot as plt
import math
# import copy
# import csv
import itertools
import sys
import copy
# from random import shuffle
from operator import itemgetter
from maze import Maze

def change(Q1, Q2, env):
	thres = 0.0 
	for i in env.free_cells:
		prev_val = sum(Q1[i[0]][i[1]])
		new_val = sum(Q2[i[0]][i[1]])
		# print (prev_val, new_val)
		if(abs(prev_val - new_val) > thres):
			change = 1
			break
		else:
			change = 0
	return change

if __name__ == "__main__":

	grid_size = 7
	# Subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ]], [[ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ]], [[ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]

	# Subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ]], [[ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]

	# random.shuffle(subtasks)
	# target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

	# target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

	grid_size = 11
	Subtasks = [[[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ]], [[ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ]], [[ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ] , [ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ] , [ 8,5 ] , [ 9,6 ] , [ 8,6 ]]]

	target_task = [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ] , [ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ] , [ 8,5 ] , [ 9,6 ] , [ 8,6 ] , [ 8,7 ] , [ 8,8 ]]

	no_tasks = len(Subtasks)+1

	print ('total no. of tasks: ',no_tasks)
	steps_per_tasks = 500
	# shuffle(subtasks)

	change_no = 5
	Rounds = 1
	curr_step = 0
	base_step = 0

	for Round in range(Rounds):
		perm = 0
		for T in itertools.permutations(Subtasks): 
			perm += 1
			if(perm != 1):
				continue
			print (perm)
			Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]
			round_no = 0
			tot_steps = 0
			subtasks = list(T)
			task_num = 0
			while (len(subtasks)):
				# print (round_no)
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
					if(task_num == 0):
						alpha = 0.4
						task_num += 1
					else:
						alpha = 1.0
					discount = 0.9
					num_actions = 4                                    # up, down, right, left
					env = Maze(grid_size, free_cells, goal)
					# if(round_no != 1):
					env.draw("curr_1/tasks/", perm, round_no, task_no)
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
								if(max(Q[curr_state[0]][curr_state[1]]) == min(Q[curr_state[0]][curr_state[1]])):
									action = -1#np.random.randint(0, num_actions)
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

					# print('Exceed: %d' % exceed)
					reward_list.append([task, tot_reward])
					tot_steps += step
				reward_list = sorted(reward_list, key = itemgetter(1))

				if(len(reward_list) == 1):
					num = 0
				else:
					num = len(reward_list)/2 + len(reward_list)%2

				subtasks = []
				# print (len(reward_list))
				# print (num)
				# print (reward_list)
				for i in range(num-1,-1,-1):
					subtasks.append(reward_list[i][0])

			# Learning of Target Task
			print ('target task')
			free_cells = target_task[:-1]
			goal = target_task[-1]

			epsilon = 0.3
			alpha = 0.999
			discount = 0.9
			num_actions = 4                                    # up, down, right, left

			env = Maze(grid_size, free_cells, goal)
			env.draw("curr_1/tasks/", perm, round_no, 5)

			tot_reward = 0
			not_change_count = 0
			r_list = [] # reward list for target task
			s_list = [] # step list for target task
			step = 0
			exceed = 0
			while True:#(step < steps_per_tasks):
				env.reset()
				game_over = False
				max_iter = 100
				itr = 0
				Q2 = copy.deepcopy(Q)
				while not (game_over or itr > max_iter):
					itr += 1
					curr_state = env.state()
					if np.random.rand() <= epsilon:
						action = np.random.randint(0, num_actions)
					else:
						if(max(Q[curr_state[0]][curr_state[1]]) == min(Q[curr_state[0]][curr_state[1]])):
							action = -1#np.random.randint(0, num_actions)
						else:
							action = np.argmax(Q[curr_state[0]][curr_state[1]])
					next_state, reward, game_over = env.act(action)
					tot_reward += reward
					step += 1
					r_list.append(tot_reward)
					s_list.append(step+tot_steps)
					# Q-learning update
					Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])
				if (itr > max_iter):
					exceed += 1
					not_change_count = 0
				elif not change(Q, Q2, env):
					not_change_count += 1
					if(not_change_count == change_no):
						break
				else:
					not_change_count = 0
					# print ('maximum steps exceeded, starting new episode.')

			# print('Exceed: %d' % exceed)
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

			# print (r_list[0], s_list[0])
			plt.figure(2)
			plt.plot(s_list,r_list)
			# plt.xscale('log')
			plt.savefig('curr_1/result_%d_2.png' % perm)

		# BASELINE

		print ('baseline')
		tot_reward = 0
		r_list = [] # reward list for target task
		s_list = [] # step list for target task
		step = 0

		epsilon = 0.3
		alpha = 0.4
		discount = 0.9
		num_actions = 4 

		Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]
		print (tot_steps)
		curr_step += tot_steps
		exceed = 0
		not_change_count = 0
		while True:#(step < tot_steps):
			env.reset()
			game_over = False
			max_iter = 100
			itr = 0
			Q2 = copy.deepcopy(Q)
			while not (game_over or itr > max_iter):
				itr += 1
				curr_state = env.state()

				if np.random.rand() <= epsilon:
					action = np.random.randint(0, num_actions)
				else:
					if(max(Q[curr_state[0]][curr_state[1]]) == min(Q[curr_state[0]][curr_state[1]])):
						action = -1#np.random.randint(0, num_actions)
					else:
						action = np.argmax(Q[curr_state[0]][curr_state[1]])
				next_state, reward, game_over = env.act(action)
				tot_reward += reward
				step += 1
				r_list.append(tot_reward)
				s_list.append(step)
				# Q-learning update
				Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])

			if (itr > max_iter):
				exceed += 1
				not_change_count = 0
			elif not change(Q, Q2, env):
				not_change_count += 1
				if(not_change_count == change_no):
					break
			else:
				not_change_count = 0
				# print ('maximum steps exceeded, starting new episode.')

		print (step)
		base_step += step

	print (curr_step/Rounds, base_step/Rounds)
	# print('Exceed: %d' % exceed)
	# plt.figure(0)
	# plt.clf()
	# plot =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
	# plt.imshow(plot, interpolation='none', cmap='gray')
	# plt.savefig("curr_1/base_policy.png")

	# plt.figure(1)
	# plt.plot(s_list, r_list)
	# # plt.xscale('log')
	# plt.yscale('log')
	# plt.savefig('curr_1/result_1.png')

	# plt.figure(2)
	# plt.plot(s_list, r_list)
	# # plt.xscale('log')
	# plt.savefig('curr_1/result_2.png')

	# print(r_list[0], s_list[0])
