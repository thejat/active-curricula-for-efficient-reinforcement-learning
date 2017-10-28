import numpy as np
from grid_world import Grid
import copy
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from scipy.spatial.distance import cosine
from numpy import linalg as LA

def change(Q1, Q2, env):
    thres = 0.0 
    prev_val = np.sum(Q1)
    new_val = np.sum(Q2)
    # print (prev_val, new_val)
    if(abs(prev_val - new_val) > thres):
        change = 1
    else:
        change = 0
    return change


def learnTask(task_num, task, Q=None, tau = -1, epsilon = 0.3, alpha = 0.6, discount = 0.9, FLAG_plot = True):
	global r_list
	global s_list
	global glob_step
	global glob_reward
	if Q == None:
		grid_size = 7
		Q = np.zeros((grid_size, grid_size, 4))
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
	return [Q, tot_reward, step]

def Transfer(Q, f_env, f_step = 500, epsilon = 0.3, alpha = 0.2, discount = 0.9):

	## Learning final task
	num_actions = f_env.num_actions
	step = 0
	tot_reward = 0
	while(step < f_step):
		f_env.reset()
		game_over = False
		max_iter = 50
		itr = 0
		while not (game_over or itr > max_iter):
			itr += 1
			curr_state = f_env.state()
			if np.random.rand() <= epsilon:
				action = np.random.randint(0, num_actions)
			else:
				if(max(Q[curr_state[0]][curr_state[1]]) == min(Q[curr_state[0]][curr_state[1]])):
					action = -1
				else:
					action = np.argmax(Q[curr_state[0]][curr_state[1]])
			next_state, reward, game_over = f_env.act(action)
			tot_reward += reward
			step += 1
			# Q-learning update
			Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])
	return [tot_reward, step]

def FindNextPair(max_eigenv, pairs, pair_features):
	max_sim = 0
	for (i,j) in pairs:
		sim = 1 - cosine(max_eigenv, pair_features[i][j])
		if(sim < 0):
			sim = -1*sim
		if( sim > max_sim):
			next_i = i
			next_j = j
			max_sim = sim
	return [next_i, next_j]

def printCurrculum(F, no_tasks):
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
	curriculum.reverse()
	print (curriculum)

if __name__ == "__main__":

	global r_list
	global s_list
	global glob_step
	global glob_reward

	grid_size = 7
	block_states = [[3,0], [3,2], [3,3], [3,4], [3,5], [1,4], [2,4], [4,3], [5,3], [6,3]]

	goal = [6,2]
	sub_start_states = [[4,1], [1,2], [0,5], [4,6]]
	dist_goal = [3, 7, 11, 16, 19]
	target_start_state = [6,5]

	subtasks = []

	for i in sub_start_states:
		Task = []
		Task.append(block_states)
		Task.append(goal)
		Task.append(i)
		subtasks.append(Task)

	target_task = []
	target_task.append(block_states)
	target_task.append(goal)
	target_task.append(target_start_state)

	tot_tasks = copy.deepcopy(subtasks)
	tot_tasks.append(target_task)

	# Evaluating task pairs features
	# sfeatures = [[16.0/6, 0, 2], [40.0/9, 1, 2], [6.0/3, 0, 1], [76.0/16, 3, 3]]
	# tfeatures = [[111.0/18, 4, 3]]
	# task_features = sfeatures + tfeatures
	# no_features = 3
	# I have 20 task pairs, I have to select some of them for learning the regression model. which one to choose?
	# Answer -- Active learning will help in it. But how many to choose?

	no_tasks = len(subtasks)+1
	pair_features = [ [[] for i in range(no_tasks)] for j in range(no_tasks-1) ]

	# for i in range(no_tasks-1):
	# 	for j in range(no_tasks):
	# 		for feat in range(no_features):
	# 			x = (task_features[i][feat] - task_features[j][feat])/max(task_features[i][feat], 0.01)
	# 			pair_features[i][j].append(x)

	# Trying another pairwise features:
	for i in range(no_tasks-1):
		for j in range(no_tasks):
			# for feat in range(no_features):
			x = 0
			for k in subtasks[i]:
				if (k in tot_tasks[j]):
					x = x+1
			if (x == 0):
				x = 0.0001
			pair_features[i][j].append(dist_goal[j]-dist_goal[i])
			pair_features[i][j].append(1)

	# print (pair_features)
	
	pair_features = np.asarray(pair_features)
	# X = []
	# y = []
	## Computing inter task tranferability matrix

	# F = np.zeros((no_tasks-1, no_tasks))
	Rounds = 30
	rounds = 20
	curr = []
	tot_F = 0
	REWARD = []#np.array([])
	STEP = []

	avg_total = 0
	total_steps_F = 0
	model_no = 0

	for Round in range(Rounds):
		r_list = np.array([])
		s_list = np.array([])
		glob_step = 0
		glob_reward = 0
		print("Round: %d" % Round)
		for next_i1 in [0]:
			for next_j1 in [2]:
				
				if(next_i1 != next_j1):
					source_learned = []
					source_Q = {}
					active_steps = [ [0 for i in range(no_tasks)] for j in range(no_tasks-1) ]
					active_pairs = []
					model_no += 1
					avg_total += 1
					X = []
					y = []
					pairs = []
					for i in range(no_tasks-1):
						for j in range(no_tasks):
							if(i != j):
								pairs.append([i, j])
					num_inputs = len(pairs)
					D = []
					F = np.zeros((no_tasks-1, no_tasks))
					next_i = next_i1
					next_j = next_j1
					print ('\n')
					print(next_i, next_j)

					ACTIVE = 6
					while len(D) != 6:
						active_pairs.append([next_i, next_j])
						pairs.remove([next_i, next_j])
						D.append(pair_features[next_i][next_j])
						X.append([])
						X[-1].append(pair_features[next_i][next_j][0])
						X[-1].append(pair_features[next_i][next_j][1])
						# X[-1].append(pair_features[next_i][next_j][2])

						if(next_i not in source_learned): 
							# s_env = Grid(grid_size, tot_tasks[next_i][0], tot_tasks[next_i][1], tot_tasks[next_i][2])
							Q, _, step = learnTask(next_i, tot_tasks[next_i])
							total_steps_F += step
							source_learned.append(next_i)
							source_Q[next_i] = [Q, step]
							active_steps[next_i][next_j] += step
						else:
							[Q, step] = source_Q[next_i]
						# print ("Task pair: (%d,%d)" % (next_i, next_j))
						# f_env = Grid(grid_size, tot_tasks[next_j][0], tot_tasks[next_j][1], tot_tasks[next_j][2])
						first = 1
						for r in range(rounds):
							Q2 = copy.deepcopy(Q)
							if first:
								first = 0
								_, F_reward, step = learnTask(next_j, tot_tasks[next_j], Q2, tau = 200)
								active_steps[next_i][next_j] += step
								total_steps_F += step
							else:
								_, F_reward, step = learnTask(next_j, tot_tasks[next_j], Q2, tau = 200, FLAG_plot = False)
							F[next_i][next_j] += F_reward
						F[next_i][next_j] /= rounds
						y.append([F[next_i][next_j], next_i, next_j])

						# print (F)

						D2 = np.asarray(D)
						A = np.dot(np.transpose(D2), D2)
						# print A
						eigenValues,eigenVectors = np.linalg.eig(A)
						idx = eigenValues.argmax()
						max_eigenv = eigenVectors[:, idx]
						[next_i, next_j] = FindNextPair(max_eigenv, pairs, pair_features)

					# print (F)
					F = F/LA.norm(F)
					# print (F)
					# printCurrculum(F, no_tasks)
					X = np.asarray(X)
					y = np.asarray(y)
					y[:,0] = y[:,0]/LA.norm(y[:,0])
					# data = np.c_[X.reshape(len(X), -1), y.reshape(len(y), -1)]
					# np.random.shuffle(data)
					# X2 = data[:, :X.size//len(X)].reshape(X.shape)
					# y2 = data[:, X.size//len(X):].reshape(y.shape)
					# [X_train, X_test] = np.split(X, [len(X)-9])
					# [y_train, y_test] = np.split(y, [len(y)-9])
					# print (len(X))
					# print (len(X_train), len(X_test))
					clf = linear_model.LinearRegression()
					# clf = linear_model.Ridge (alpha = .1)
					clf.fit(X, y[:,0])
					# predict_test = clf.predict(X_test)
					# print (predict_test, y_test[:,0])

					# plt.figure(3)
					# plt.clf()
					# plt.scatter(X[:,0], y[:,0], color='red')
					# predict_train = clf.predict(X_train)
					# plt.scatter(X_train[:,0], predict_train, color='blue')
					# plt.scatter(X_test[:,0], predict_test, color='magenta')

					Xt = []
					non_pairs = []
					for i in range(no_tasks-1):
						for j in range(no_tasks):
							if [i,j] not in active_pairs:
								Xt.append([])
								Xt[-1].append(pair_features[i][j][0])
								Xt[-1].append(pair_features[i][j][1])
								non_pairs.append([i,j])
					predict = clf.predict(Xt)
					# for i in range(len(X)):
					# 	plt.plot([X[i][0], X[i][0]], [y[i][0], predict[i]], color='green')
					# plt.savefig('active/model1_%d.png' % model_no)

					# plt.figure(2)
					# plt.imshow(F, interpolation='none', cmap='gray')
					# plt.savefig('active_F_model/F.png')

					for i in range(len(Xt)):
						Ts = int(non_pairs[i][0])
						Tf = int(non_pairs[i][1])
						F[Ts][Tf] = predict[i]

					# print (F)
					# plt.figure(2)
					# plt.imshow(F, interpolation='none', cmap='gray')
					# plt.savefig('active_F_model/active_F.png')
					printCurrculum(F, no_tasks)
					# print (total_steps_F/avg_total)
					tot_active_steps = 0
					for i in range(len(X)):
						# print (active_pairs[i][0], active_pairs[i][1])
						# print (active_steps[active_pairs[i][0]][active_pairs[i][1]])
						tot_active_steps += active_steps[active_pairs[i][0]][active_pairs[i][1]]
					print (tot_active_steps)

		print r_list[-1], s_list[-1]

		REWARD.append(r_list)
		STEP.append(s_list)

	REWARD = np.asarray(REWARD)
	STEP = np.asarray(STEP)
	np.savez("active", REWARD=REWARD, STEP=STEP)



