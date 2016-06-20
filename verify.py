import numpy as np
import matplotlib.pyplot as plt
from maze import Maze
import itertools
import copy

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

def learntask(task, Q, grid_size, epsilon = 0.3, alpha = 0.6, discount = 0.9):
	env = Maze(grid_size, task[:-1], task[-1])
	num_actions = env.num_actions
	step = 0
	episode = 0
	exceed = 0
	not_change_count = 0
	change_no = 5
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
	print ('Exceed: %d, Episode: %d' %(exceed, episode))
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

	print('plotting')
	plt.figure(1)
	plt.plot(s_list, r_list)
	# plt.xscale('log')
	# plt.yscale('log')
	plt.savefig('verify/resutl%d_1.png' % perm)
	print ('done')
	print('plotting')
	plt.figure(2)
	plt.plot(s_list, r_list)
	plt.yscale('log')
	plt.savefig('verify/resutl%d_2.png' % perm)
	print ('done')

	print ('Exceed: %d, Episode: %d' %(exceed, episode))

	return Q

def plot_policy(num, Q):
	plt.figure(3)
	plt.clf()
	grid_size = len(Q)
	plot =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
	plt.imshow(plot, interpolation='none', cmap='gray')
	plt.savefig("verify/%d_policy.png" % (num))

if __name__ == "__main__":

	subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ]], [[ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]

	target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

	no_tasks = len(subtasks)+1
	grid_size = 7

	perm = 0

	for T in itertools.permutations(subtasks):
		perm += 1
		print ('perm no. %d' % perm)
		Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]
		tot_step = 0
		for task in T:
			[Q, step] = learntask(task, Q, grid_size)
			tot_step += step

		Q = learnTtask(perm, target_task, Q, grid_size, tot_step)
		plot_policy(perm, Q)

	# BASELINE
	print ('baseline')
	Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]
	Q = learnTtask(0, target_task, Q, grid_size, 0)
	plot_policy(0, Q)