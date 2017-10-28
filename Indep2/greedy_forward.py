import numpy as np
from grid_world import Grid
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

def learnTask(task_num, Q, task, tau = -1, epsilon = 0.3, alpha = 0.6, discount = 0.9):
	global r_list
	global s_list
	global glob_step
	global glob_reward
	grid_size = len(Q)
	env = Grid(grid_size, task[0], task[1], task[2])
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
		max_iter = 500
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
			glob_reward += reward
			glob_step += 1
			step += 1
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
	return Q

def EvalTask(Q, task, tau = 200, epsilon = 0.3, alpha = 0.6, discount = 0.9):
	global r_list
	global s_list
	global glob_step
	global glob_reward
	repeat_no = 1
	env = Grid(grid_size, task[0], task[1], task[2])
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
					if(np.amax(Q[curr_state[0]][curr_state[1]]) == np.amin(Q[curr_state[0]][curr_state[1]])):
						action = np.random.randint(0, num_actions)
					else:
						action = np.argmax(Q[curr_state[0]][curr_state[1]])
				next_state, reward, game_over = env.act(action)
				tot_reward += reward
				glob_reward += reward
				glob_step += 1
				step += 1
				r_list = np.append(r_list, glob_reward)
				s_list = np.append(s_list, glob_step)
				# Q-learning update
				Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*np.amax(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])
	return 1.0*tot_reward/repeat_no

def findTask(Q, subtasks):
	max_reward = -1
	for task in subtasks:
		Q2 = copy.deepcopy(Q)
		reward = EvalTask(Q2, task)
		if(reward > max_reward):
			max_reward = reward
			retTask = task
	return retTask

if __name__ == "__main__":

	global r_list
	global s_list
	global glob_step
	global glob_reward

	grid_size = 7
	block_states = [[3,0], [3,2], [3,3], [3,4], [3,5], [1,4], [2,4], [4,3], [5,3], [6,3]]

	goal = [6,2]
	sub_start_states = [[4,1], [1,2], [0,5], [4,6]]
	target_start_state = [6,5]

	Subtasks = []

	for i in sub_start_states:
		Task = []
		Task.append(block_states)
		Task.append(goal)
		Task.append(i)
		Subtasks.append(Task)
	target_task = []
	target_task.append(block_states)
	target_task.append(goal)
	target_task.append(target_start_state)  

	Rounds = 30
	REWARD = []#np.array([])
	STEP = []#np.array([])
	BREWARD = []#np.array([])
	BSTEP = []#np.array([])
	for Round in range(Rounds):
		print ("Round no. %d" % Round)
		subtasks = copy.deepcopy(Subtasks)
		# Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]
		Q = np.zeros((grid_size, grid_size, 4))
		# tot_step = 0
		task_num = 1
		# first_task = 1
		curr = []
		r_list = np.array([])
		s_list = np.array([])
		glob_step = 0
		glob_reward = 0
		while len(subtasks) != 0:
			task = findTask(Q, subtasks)
			curr.append(Subtasks.index(task))
			Q = learnTask(task_num, Q, task, alpha = 0.6)
			subtasks.remove(task)
			task_num += 1

		print(curr)
		Q = learnTask(5, Q, target_task, alpha = 0.6)

		print ("curr: steps -- %d reward -- %d" %(s_list[-1],r_list[-1]))
		REWARD.append(r_list)
		STEP.append(s_list)
		
		r_list = np.array([])
		s_list = np.array([])
		glob_step = 0
		glob_reward = 0

		Q = np.zeros((grid_size, grid_size, 4))
		Q = learnTask(0, Q, target_task, alpha = 0.6)
		print ("base: steps -- %d reward -- %d" %(s_list[-1],r_list[-1]))
		BREWARD.append(r_list)
		BSTEP.append(s_list)

	REWARD = np.asarray(REWARD)
	STEP = np.asarray(STEP)

	np.savez("greedy", REWARD=REWARD, STEP=STEP)
	BREWARD = np.asarray(BREWARD)
	BSTEP = np.asarray(BSTEP)
	np.savez("base", BREWARD=BREWARD, BSTEP=BSTEP)
	
	# print (curr_step/Rounds)
	# print (base_step/Rounds)


	# grid_size = 7
	# subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ]], [[ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ]], [[ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]
	# Subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 4,0 ]], [[ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]
	# random.shuffle(subtasks)
	# target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]
	# target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

	# grid_size = 11
	# Subtasks = [[[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ]], [[ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ]], [[ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ] , [ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ] , [ 8,5 ] , [ 9,6 ] , [ 8,6 ]]]
	# target_task = [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ] , [ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ] , [ 8,5 ] , [ 9,6 ] , [ 8,6 ] , [ 8,7 ] , [ 8,8 ]]

# def plotlearnTask(Q, task, tau = -1, epsilon = 0.3, alpha = 0.6, discount = 0.9):
# 	grid_size = len(Q)
# 	env = Maze(grid_size, task[:-1], task[-1])
# 	# env.draw("greedy_max/tasks/", task_num)
# 	num_actions = env.num_actions
# 	## Learning source task
# 	episode = 0
# 	exceed = 0
# 	not_change_count = 0
# 	change_no = 5
# 	step = 0
# 	while ((True and tau == -1) or step < tau):
# 		env.reset()
# 		game_over = False
# 		max_iter = 100
# 		itr = 0
# 		episode += 1
# 		Q2 = copy.deepcopy(Q)
# 		while not (game_over or itr > max_iter):
# 			itr += 1
# 			curr_state = env.state()
# 			if np.random.rand() <= epsilon:
# 				action = np.random.randint(0, num_actions)
# 			else:
# 				if(np.amax(Q[curr_state[0]][curr_state[1]]) == np.amin(Q[curr_state[0]][curr_state[1]])):
# 					action = -1#np.random.randint(0, num_actions)
# 				else:
# 					action = np.argmax(Q[curr_state[0]][curr_state[1]])
# 			next_state, reward, game_over = env.act(action)
# 			glob_reward += reward
# 			glob_step += 1
# 			r_list.append(glob_reward)
# 			s_list.append(glob_step)
# 			# Q-learning update
# 			Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*np.amax(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])
# 		if (itr > max_iter):
# 			exceed += 1
# 			not_change_count = 0
# 		elif not change(Q, Q2, env):
# 			not_change_count += 1
# 			if(not_change_count == change_no):
# 				break
# 		else:
# 			not_change_count = 0
	# print ('Exceed: %d/%d' % (exceed, episode))
	# policy =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
	# plt.figure(1)
	# plt.clf()
	# plt.imshow(policy, interpolation='none', cmap='gray')
	# if first_step:
	# 	first_step = 1
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

	# return step