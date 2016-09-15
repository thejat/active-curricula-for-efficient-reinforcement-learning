import numpy as np
import matplotlib.pyplot as plt
from maze import Maze
import itertools
import copy
import csv

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

def learntask(perm, task_num, task, Q, grid_size, epsilon = 0.3, alpha = 0.6, discount = 0.9, tau = 500):
	env = Maze(grid_size, task[:-1], task[-1])
	# if(task_num == 1):
	# 	env.draw("verify2/tasks/", perm, task_num)
	num_actions = env.num_actions
	step = 0
	episode = 0
	exceed = 0
	not_change_count = 0
	change_no = 5
	while step < tau:
		env.reset()
		game_over = False
		max_iter = 100
		itr = 0
		episode += 1
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
			step += 1
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
	# print ('Exceed: %d, Episode: %d' %(exceed, episode))
	return [Q, step]

def learnTtask(perm, task, Q, grid_size, tot_step, epsilon = 0.3, alpha = 0.6, discount = 0.9):
	env = Maze(grid_size, task[:-1], task[-1])
	num_actions = env.num_actions
	step = tot_step
	episode = 0
	exceed = 0
	not_change_count = 0
	change_no = 5
	tot_reward = 0
	s_list = []
	r_list = []
	while True:
		env.reset()
		game_over = False
		max_iter = 100
		itr = 0
		episode += 1
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

	# print('plotting')
	# plt.figure(1)
	# plt.plot(s_list, r_list)
	# # plt.xscale('log')
	# # plt.yscale('log')
	# plt.savefig('verify/resutl%d_1.png' % perm)
	# print ('done')
	# print('plotting')
	# plt.figure(2)
	# plt.plot(s_list, r_list)
	# plt.yscale('log')
	# plt.savefig('verify/resutl%d_2.png' % perm)
	# print ('done')

	# print ('Exceed: %d, Episode: %d' %(exceed, episode))

	return [Q, step]

def plot_policy(num, Q):
	plt.figure(3)
	plt.clf()
	grid_size = len(Q)
	plot =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
	plt.imshow(plot, interpolation='none', cmap='gray')
	plt.savefig("verify2/policies/%d_policy.png" % (num))

def fact(n):
	if(n == 1):
		return 1
	else:
		return n*fact(n-1)

if __name__ == "__main__":

	# subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ]], [[ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]

	# target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

	# subtasks = [[[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ]], [[ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ]], [[ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ] , [ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ] , [ 8,5 ] , [ 9,6 ] , [ 8,6 ]]]

	# target_task = [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ] , [ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ] , [ 8,5 ] , [ 9,6 ] , [ 8,6 ] , [ 8,7 ] , [ 8,8 ]]
	
	# changed july 15:
	subtasks = [[[ 4,0 ] , [ 4,1 ], [ 4,2 ] , [ 4,3 ] , [ 4,4 ]], [[ 4,0 ] , [ 4,1 ] , [ 5,2 ] , [ 6,2] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]], [[ 4,0 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]], [[ 0, 0 ],  [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]]

	target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

	no_tasks = len(subtasks)+1
	grid_size = 7
	print (no_tasks)
	all_steps = [0 for i in range(fact(no_tasks-1)+1)]
	print (len(all_steps))
	Rounds = 50
	for Round in range(Rounds):
		print("Round no. %d" % (Round+1))
		perm = 0
		ind = 0
		for T in itertools.permutations(subtasks):
			perm += 1
			if(perm%5000 == 0):
				print(perm)
			if(perm%1 == 0):
				# print ('perm no. %d' % perm)
				Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]
				tot_step = 0
				task_num = 0
				for task in T:
					task_num += 1
					if(task_num == 1):
						[Q, step] = learntask(perm, task_num, task, Q, grid_size, alpha = 0.2)
						# print (step)
					else:
						[Q, step] = learntask(perm, task_num, task, Q, grid_size, alpha = 0.2)
					tot_step += step
				# print(tot_step)
				[Q, tot_step] = learnTtask(perm, target_task, Q, grid_size, tot_step, alpha = 0.2)
				# print(tot_step)
				all_steps[ind] += tot_step
				ind += 1
				# print(tot_step)
			
				# plot_policy(perm, Q)

		# BASELINE
		print ('baseline')
		Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]
		[Q, tot_step] = learnTtask(0, target_task, Q, grid_size, 0, alpha = 0.2)
		all_steps[ind] += tot_step
		print (all_steps[0:ind+1])
		with open("verify_new_subtasks.csv", "w") as fp:
		    wr = csv.writer(fp)
		    for i in all_steps:
		    	wr.writerow([i/(Round+1)])
		# plot_policy(0, Q)

	# all_steps[:ind+1] = [x/Rounds for x in all_steps[0:ind+1]] 
	# with open("verify2.csv", "a") as fp:
	#     wr = csv.writer(fp)
	#     wr.writerow(all_steps[0:ind+1])
	# print (all_steps[0:ind+1])
	
