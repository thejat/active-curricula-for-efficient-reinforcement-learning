import numpy as np
from maze import Maze 
import copy
import random
import matplotlib.pyplot as plt

def learn_source(task_num, s_env, s_step = 30000, epsilon = 0.3, alpha = 0.6, discount = 0.9):

	grid_size = s_env.grid_size
	Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]
	num_actions = s_env.num_actions

	## Learning source task
	step = 0
	episode = 0
	exceed = 0
	avg_step = 0
	while(step < s_step):
		s_env.reset()
		game_over = False
		max_iter = 100
		itr = 0
		episode += 1
		while not (game_over or itr > max_iter):
			itr += 1
			curr_state = s_env.state()
			if np.random.rand() <= epsilon:
				action = np.random.randint(0, num_actions)
			else:
				action = np.argmax(Q[curr_state[0]][curr_state[1]])
			next_state, reward, game_over = s_env.act(action)
			step += 1
			# Q-learning update
			Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])
		if(itr > max_iter):
			exceed += 1
		else:
			# print (itr)
			avg_step += itr
			# print ('maximum steps exceeded, starting new episode.')

	print ('Exceed: %d/%d' % (exceed, episode))
	print ('Avg step: %d, Episodes: %d' % ((avg_step/(episode - exceed)), episode))
	policy =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
	plt.figure(1)
	plt.clf()
	plt.imshow(policy, interpolation='none', cmap='gray')
	plt.savefig("inter/policies/source_policy%d.png" % task_num)

	return Q

def Transfer(Q, f_env, f_step = 8000, epsilon = 0.3, alpha = 0.6, discount = 0.9):

	## Learning final task
	num_actions = f_env.num_actions
	step = 0
	tot_reward = 0
	while(step < f_step):
		f_env.reset()
		game_over = False
		max_iter = 100
		itr = 0
		while not (game_over or itr > max_iter):
			itr += 1
			curr_state = f_env.state()
			if np.random.rand() <= epsilon:
				action = np.random.randint(0, num_actions)
			else:
				action = np.argmax(Q[curr_state[0]][curr_state[1]])
			next_state, reward, game_over = f_env.act(action)
			tot_reward += reward
			step += 1
			# Q-learning update
			Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])
		# if(itr > max_iter):
		# 	print ('maximum steps exceeded, starting new episode.')
	print (tot_reward)
	return tot_reward

if __name__ == "__main__":

	gridsize = 7
	# subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ]], [[ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ]], [[ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]

	subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ]], [[ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]

	# random.shuffle(subtasks)
	# target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

	target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

	tot_tasks = copy.deepcopy(subtasks)
	tot_tasks.append(target_task)

	no_tasks = len(tot_tasks)

	F = np.zeros((no_tasks-1, no_tasks))

	for Ts in range(no_tasks-1):
		s_env = Maze(gridsize, tot_tasks[Ts][:-1], tot_tasks[Ts][-1])
		s_env.draw(Ts)
		Q = learn_source(Ts, s_env)
		for Tf in range(no_tasks):
			if(Ts == Tf):
				F[Ts][Tf] = -1000
			else:
				print ("Task pair: (%d,%d)" % (Ts, Tf))
				f_env = Maze(gridsize, tot_tasks[Tf][:-1], tot_tasks[Tf][-1])
				# print Q
				Q2 = copy.deepcopy(Q)
				F[Ts][Tf] = Transfer(Q2, f_env)

	plt.figure(2)
	plt.imshow(F, interpolation='none', cmap='gray')
	plt.savefig('inter/F.png')
	print ("F matrix:")
	print (F)

	## Assuming target task is given, we will now find a curriculum using F matrix

	curriculum = []
	last_task = no_tasks - 1 #index of target task
	while True:
		curriculum.append(last_task)
		F_val = -10001
		for i in range(no_tasks-1):
			if(F_val < F[i][last_task] and i not in curriculum):
				F_val = F[i][last_task]
				next_task = i
		if (next_task in curriculum):
			break;
		else:
			last_task = next_task

	print ("Curriculum:")
	print (curriculum)
