import numpy as np
from maze import Maze 
from operator import itemgetter
import copy
import random
import matplotlib.pyplot as plt

r_list = np.array([])
s_list = np.array([])

glob_step = 0
glob_reward = 0

def change(Q1, Q2, env):
    thres = 0.0
    prev_val = np.sum(Q1)
    new_val = np.sum(Q2)
    if(abs(prev_val - new_val) > thres):
        change = 1
    else:
        change = 0
    return change

def learnTask(task_num, Q, task, tau = -1, epsilon = 0.3, alpha = 0.6, discount = 0.9, FLAG_plot = True):
	global r_list
	global s_list
	global glob_step
	global glob_reward
	grid_size = len(Q)
	env = Maze(grid_size, task[:-1], task[-1])
	num_actions = env.num_actions
	## Learning source task
	step = 0
	episode = 0
	exceed = 0
	not_change_count = 0
	change_no = 5
	tot_reward = 0;
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
				if(np.amax(Q[curr_state[0]][curr_state[1]]) == np.amin(Q[curr_state[0]][curr_state[1]])):
					action = -1#np.random.randint(0, num_actions)
				else:
					action = np.argmax(Q[curr_state[0]][curr_state[1]])
			next_state, reward, game_over = env.act(action)
			step += 1
			tot_reward += reward
			if FLAG_plot:
				glob_reward += reward
				glob_step += 1
				r_list = np.append(r_list, glob_reward)
				s_list = np.append(s_list, glob_step)
			# Q-learning update
			Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*np.amax(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])
		if (itr > max_iter):
			exceed += 1
			not_change_count = 0
		elif not change(Q, Q2, env):
			not_change_count += 1
			if(not_change_count == change_no):
				break
		else:
			not_change_count = 0
	# print step
	return [Q, reward]

if __name__ == "__main__":

	global r_list
	global s_list
	global glob_step
	global glob_reward

	grid_size = 7
	subtasks = [[[ 4,0 ] , [ 4,1 ], [ 4,2 ] , [ 4,3 ] , [ 4,4 ]], [[ 4,0 ] , [ 4,1 ] , [ 5,2 ] , [ 6,2] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]], [[ 4,0 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]], [[ 0, 0 ],  [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]]

	target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]


	tot_tasks = copy.deepcopy(subtasks)
	tot_tasks.append(target_task)

	no_tasks = len(tot_tasks)

	Rounds = 30
	rounds = 20
	curr = []
	tot_F = 0
	REWARD = []#np.array([])
	STEP = []
	for Round in range(Rounds):
		r_list = np.array([])
		s_list = np.array([])
		glob_step = 0
		glob_reward = 0
		print("Round: %d" % Round)
		F = np.zeros((no_tasks-1, no_tasks))
		source_step = [0 for i in range(no_tasks-1)]
		for Ts in range(no_tasks-1):
			Q = np.zeros((grid_size, grid_size, 4))
			[Q, reward] = learnTask(0, Q, tot_tasks[Ts])
			for Tf in range(no_tasks):
				if(Ts == Tf):
					F[Ts][Tf] = -1000
				else:
					F[Ts][Tf] = 0 
					first = 1
					for r in range(rounds):
						Q2 = copy.deepcopy(Q)
						if first:
							first = 0
							[Q2, F_reward] = learnTask(0, Q2, tot_tasks[Tf], tau=100)
						else:
							[Q2, F_reward] = learnTask(0, Q2, tot_tasks[Tf], tau=100, FLAG_plot = False)
						F[Ts][Tf] += F_reward 

					F[Ts][Tf] /= rounds 

		print ('Total steps required to calculate F matrix: %d' %s_list[-1])
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

		curriculum.reverse()
		print "Curriculum:", curriculum
		curr.append(curriculum)
		REWARD.append(r_list)
		STEP.append(s_list)

	# print(tot_F)
	print(curr)

	REWARD = np.asarray(REWARD)
	STEP = np.asarray(STEP)
	np.savez("inter", REWARD=REWARD, STEP=STEP)

	unique = []
	for x in curr:
		if x not in unique:
			unique.append(x)
	for x in unique:
		print(x, curr.count(x))

# a = [[2, 1, 0, 3, 4], [2, 1, 3, 0, 4], [2, 1, 3, 0, 4], [2, 0, 1, 3, 4], [0, 2, 1, 3, 4], [2, 0, 1, 3, 4], [2, 1, 0, 3, 4], [0, 1, 2, 3, 4], [1, 2, 3, 0, 4], [0, 2, 1, 3, 4], [0, 2, 1, 3, 4], [2, 1, 0, 3, 4], [2, 1, 0, 3, 4], [1, 0, 2, 3, 4], [0, 1, 2, 3, 4], [2, 1, 3, 0, 4], [2, 1, 3, 0, 4], [0, 2, 1, 3, 4], [2, 1, 3, 0, 4], [2, 1, 3, 0, 4], [0, 1, 3, 2, 4], [2, 0, 3, 1, 4], [2, 1, 0, 3, 4], [2, 0, 1, 3, 4], [0, 2, 1, 3, 4], [2, 0, 1, 3, 4], [0, 2, 3, 1, 4], [0, 1, 3, 2, 4], [2, 3, 1, 0, 4], [2, 3, 1, 0, 4], [2, 0, 1, 3, 4], [2, 1, 0, 3, 4], [0, 2, 3, 1, 4], [2, 3, 1, 0, 4], [3, 1, 0, 2, 4], [0, 2, 1, 3, 4], [0, 1, 3, 2, 4], [0, 2, 1, 3, 4], [1, 0, 3, 2, 4], [0, 2, 3, 1, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [1, 2, 3, 0, 4], [2, 0, 1, 3, 4], [2, 0, 1, 3, 4], [2, 3, 1, 0, 4], [0, 2, 3, 1, 4], [2, 3, 1, 0, 4], [2, 0, 3, 1, 4], [1, 0, 3, 2, 4]]


	# gridsize = 7
	# subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ]], [[ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ]], [[ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]

	# subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ]], [[ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]

	# random.shuffle(subtasks)
	# target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

	# target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

	# gridsize = 11
	# subtasks = [[[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ]], [[ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ]], [[ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ] , [ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ] , [ 8,5 ] , [ 9,6 ] , [ 8,6 ]]]

	# target_task = [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ] , [ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ] , [ 8,5 ] , [ 9,6 ] , [ 8,6 ] , [ 8,7 ] , [ 8,8 ]]

# def change(Q1, Q2, env):
# 	thres = 0.0 
# 	for i in env.free_cells:
# 		prev_val = sum(Q1[i[0]][i[1]])
# 		new_val = sum(Q2[i[0]][i[1]])
# 		# print (prev_val, new_val)
# 		if(abs(prev_val - new_val) > thres):
# 			change = 1
# 			break
# 		else:
# 			change = 0
# 	return change

# def learn_source(task_num, s_env, epsilon = 0.3, alpha = 0.2, discount = 0.9):

# 	grid_size = s_env.grid_size
# 	Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]
# 	num_actions = s_env.num_actions
# 	## Learning source task
# 	step = 0
# 	episode = 0
# 	exceed = 0
# 	not_change_count = 0
# 	change_no = 5
# 	while True:
# 		s_env.reset()
# 		game_over = False
# 		max_iter = 100
# 		itr = 0
# 		episode += 1
# 		Q2 = copy.deepcopy(Q)
# 		while not (game_over or itr > max_iter):
# 			itr += 1
# 			curr_state = s_env.state()
# 			if np.random.rand() <= epsilon:
# 				action = np.random.randint(0, num_actions)
# 			else:
# 				if(max(Q[curr_state[0]][curr_state[1]]) == min(Q[curr_state[0]][curr_state[1]])):
# 					action = -1#np.random.randint(0, num_actions)
# 				else:
# 					action = np.argmax(Q[curr_state[0]][curr_state[1]])
# 			next_state, reward, game_over = s_env.act(action)
# 			step += 1
# 			# Q-learning update
# 			Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])
# 		if (itr > max_iter):
# 			exceed += 1
# 			not_change_count = 0
# 		elif not change(Q, Q2, s_env):
# 			not_change_count += 1
# 			if(not_change_count == change_no):
# 				break
# 		else:
# 			not_change_count = 0
# 	# print ('Exceed: %d/%d' % (exceed, episode))
# 	# print (step)
# 	# policy =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
# 	# plt.figure(1)
# 	# plt.clf()
# 	# plt.imshow(policy, interpolation='none', cmap='gray')
# 	# plt.savefig("inter/policies/source_policy%d.png" % task_num)
# 	return [Q, step]

# par = 0
# def Transfer2(Q, f_env, epsilon = 0.1, alpha = 0.2, discount = 0.9): 
# 	## Learning final task
# 	num_actions = f_env.num_actions
# 	step = 0
# 	tot_reward = 0
# 	episode = 0
# 	exceed = 0
# 	not_change_count = 0
# 	change_no = 10
# 	while True:
# 		f_env.reset()
# 		game_over = False
# 		max_iter = 100
# 		itr = 0
# 		episode += 1
# 		Q2 = copy.deepcopy(Q)
# 		while not (game_over or itr > max_iter):
# 			itr += 1
# 			curr_state = f_env.state()
# 			if np.random.rand() <= epsilon:
# 				action = np.random.randint(0, num_actions)
# 			else:
# 				if(max(Q[curr_state[0]][curr_state[1]]) == min(Q[curr_state[0]][curr_state[1]])):
# 					action = -1#np.random.randint(0, num_actions)
# 				else:
# 					action = np.argmax(Q[curr_state[0]][curr_state[1]])
# 			next_state, reward, game_over = f_env.act(action)
# 			tot_reward += reward
# 			step += 1
# 			# Q-learning update
# 			Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])
# 		if (itr > max_iter):
# 			exceed += 1
# 			not_change_count = 0
# 		elif not change(Q, Q2, s_env):
# 			not_change_count += 1
# 			if(not_change_count == change_no):
# 				break
# 		else:
# 			not_change_count = 0

# 	global par
# 	# policy =  [[max(Q[i][j]) for i in range(len(Q))] for j in range(len(Q))]
# 	# plt.figure(1)
# 	# plt.clf()
# 	# plt.imshow(policy, interpolation='none', cmap='gray')
# 	# plt.savefig("inter/policies/target_policy%d.png" % par)
# 	par += 1
# 	print (tot_reward, step)
# 	return 1.0/step

# def Transfer(Q, f_env, f_step = 500, epsilon = 0.3, alpha = 0.2, discount = 0.9):

# 	## Learning final task
# 	num_actions = f_env.num_actions
# 	step = 0
# 	tot_reward = 0
# 	while(step < f_step):
# 		f_env.reset()
# 		game_over = False
# 		max_iter = 50
# 		itr = 0
# 		while not (game_over or itr > max_iter):
# 			itr += 1
# 			curr_state = f_env.state()
# 			if np.random.rand() <= epsilon:
# 				action = np.random.randint(0, num_actions)
# 			else:
# 				if(max(Q[curr_state[0]][curr_state[1]]) == min(Q[curr_state[0]][curr_state[1]])):
# 					action = -1#np.random.randint(0, num_actions)
# 				else:
# 					if(max(Q[curr_state[0]][curr_state[1]]) == min(Q[curr_state[0]][curr_state[1]])):
# 						action = -1#np.random.randint(0, num_actions)
# 					else:
# 						action = np.argmax(Q[curr_state[0]][curr_state[1]])
# 			next_state, reward, game_over = f_env.act(action)
# 			tot_reward += reward
# 			step += 1
# 			# Q-learning update
# 			Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])
# 	# print (tot_reward)
# 	return [tot_reward, step]
