import numpy as np
from maze import Maze 
import copy
import random
import matplotlib.pyplot as plt

def change(Q1, Q2, env):
	thres = 0.0 
	for i in env.free_cells:
		prev_val = sum(Q1[i[0]][i[1]])
		new_val = sum(Q2[i[0]][i[1]])
		if(abs(prev_val - new_val) > thres):
			change = 1
			break
		else:
			change = 0
	return change

def learnTask(task_num, Q, task, tau = -1, epsilon = 0.3, alpha = 0.6, discount = 0.9):
	grid_size = len(Q)
	env = Maze(grid_size, task[:-1], task[-1])
	# env.draw("greedy_max/tasks/", task_num)
	num_actions = env.num_actions
	## Learning source task
	step = 0
	episode = 0
	exceed = 0
	not_change_count = 0
	change_no = 5
	while ((True and tau == -1) or step < tau):
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
	# print ('Exceed: %d/%d' % (exceed, episode))
	# policy =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
	# plt.figure(1)
	# plt.clf()
	# plt.imshow(policy, interpolation='none', cmap='gray')
	# plt.savefig("greedy_max/policies/source_policy%d.png" % task_num)
	return [Q, step]

def plotlearnTask(Q, task, first_step, tau = -1, epsilon = 0.3, alpha = 0.6, discount = 0.9):
	grid_size = len(Q)
	env = Maze(grid_size, task[:-1], task[-1])
	# env.draw("greedy_max/tasks/", task_num)
	num_actions = env.num_actions
	## Learning source task
	step = first_step
	episode = 0
	exceed = 0
	not_change_count = 0
	change_no = 5
	r_list = []
	s_list = []
	tot_reward = 0
	while ((True and tau == -1) or step < tau):
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
	# print ('Exceed: %d/%d' % (exceed, episode))
	policy =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
	plt.figure(1)
	plt.clf()
	plt.imshow(policy, interpolation='none', cmap='gray')
	if first_step:
		first_step = 1
	# plt.savefig("greedy_max/policies/target_policy%d.png" % first_step)

	# print('plotting')
	# plt.figure(2)
	# plt.plot(s_list, r_list)
	# # plt.xscale('log')
	# # plt.yscale('log')
	# plt.savefig('greedy_max/resutl_1.png')
	# print ('done')
	# print('plotting')
	# plt.figure(3)
	# plt.plot(s_list, r_list)
	# plt.yscale('log')
	# plt.savefig('greedy_max/resutl_2.png')
	# print ('done')

	return step

def EvalTask(Q, task, tau = 500, epsilon = 0.3, alpha = 1.0, discount = 0.9):
	repeat_no = 1
	env = Maze(len(Q), task[:-1], task[-1])
	num_actions = env.num_actions
	step = 0
	tot_reward = 0
	for i in range(repeat_no):
		while(step < tau):
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
						action = np.random.randint(0, num_actions)
					else:
						action = np.argmax(Q[curr_state[0]][curr_state[1]])
				next_state, reward, game_over = env.act(action)
				tot_reward += reward
				step += 1
				# Q-learning update
				Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])
	return [tot_reward/repeat_no, step/repeat_no]

def findTask(Q, subtasks):
	reward_list = []
	max_reward = -1
	tot_step = 0
	for task in subtasks:
		Q2 = copy.deepcopy(Q)
		[reward, step] = EvalTask(Q2, task)
		tot_step += step
		reward_list.append(reward)
		if(reward > max_reward):
			max_reward = reward
			retTask = task
	# print (reward_list)
	return [retTask, tot_step]

if __name__ == "__main__":

	# grid_size = 7
	# subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ]], [[ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ]], [[ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]
	# Subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 4,0 ]], [[ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]
	# random.shuffle(subtasks)
	# target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]
	# target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

	grid_size = 11
	Subtasks = [[[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ]], [[ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ]], [[ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ] , [ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ] , [ 8,5 ] , [ 9,6 ] , [ 8,6 ]]]

	target_task = [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ] , [ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ] , [ 8,5 ] , [ 9,6 ] , [ 8,6 ] , [ 8,7 ] , [ 8,8 ]]

	Rounds = 50
	curr_step = 0
	base_step = 0
	for Round in range(Rounds):
		print ("Round no. %d" % Round)
		subtasks = copy.deepcopy(Subtasks)
		Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]
		tot_step = 0
		task_num = 1
		first_task = 1
		curr = []
		while len(subtasks) != 0:
			[task, tstep] = findTask(Q, subtasks)
			print ('eval: %d' % tstep)
			curr.append(Subtasks.index(task))
			tot_step += tstep
			if(first_task):
				[Q, step] = learnTask(task_num, Q, task, alpha = 0.4)
				first_task = 0
			else:
				[Q, step] = learnTask(task_num, Q, task, alpha = 1.0)
			tot_step += step
			subtasks.remove(task)
			task_num += 1

		print(curr)
		step = plotlearnTask(Q, target_task, tot_step, alpha = 1.0)
		tot_step = step
		print ("curr: %d" %tot_step)
		curr_step += tot_step

		Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]
		step2 = plotlearnTask(Q, target_task, 0, alpha = 0.4)
		print ("base: %d" %step2)
		base_step += step2

	print (curr_step/Rounds)
	print (base_step/Rounds)