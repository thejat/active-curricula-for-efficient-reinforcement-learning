import numpy as np
from grid_world import Grid 
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
	tot_reward = 0;
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
			tot_reward += reward
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
	return [Q, reward]

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

	for Round in range(Rounds):
		r_list = np.array([])
		s_list = np.array([])
		glob_step = 0
		glob_reward = 0
		subtasks = copy.deepcopy(Subtasks)
		Q = np.zeros((grid_size, grid_size, 4))
		task_num = 0
		while(len(subtasks)):
			print len(subtasks)
			reward_list = []
			for task in subtasks:
				task_num += 1
				[Q, reward] = learnTask(task_num, Q, task, tau=500)
				reward_list.append([task, reward])
				reward_list = sorted(reward_list, key = itemgetter(1))
				if(len(reward_list) == 1):
					num = 0
				else:
					num = int(len(reward_list)/2) + len(reward_list)%2
				subtasks = []
				for i in range(num-1,-1,-1):
					subtasks.append(reward_list[i][0])
		learnTask(0, Q, target_task)
		print ("curr: steps -- %d reward -- %d" %(s_list[-1],r_list[-1]))
		REWARD.append(r_list)
		STEP.append(s_list)

	REWARD = np.asarray(REWARD)
	STEP = np.asarray(STEP)
	np.savez("bc", REWARD=REWARD, STEP=STEP)