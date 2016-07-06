import numpy as np
from maze import Maze 
import copy
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from scipy.spatial.distance import cosine
from numpy import linalg as LA

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

def learn_source(task_num, s_env, epsilon = 0.3, alpha = 0.6, discount = 0.9):

	grid_size = s_env.grid_size
	Q = [[[0,0,0,0] for i in range(grid_size)] for j in range(grid_size)]
	num_actions = s_env.num_actions

	## Learning source task
	step = 0
	episode = 0
	exceed = 0
	not_change_count = 0
	change_no = 5
	while True:
		s_env.reset()
		game_over = False
		max_iter = 100
		itr = 0
		episode += 1
		Q2 = copy.deepcopy(Q)
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
		if (itr > max_iter):
			exceed += 1
			not_change_count = 0
		elif not change(Q, Q2, s_env):
			not_change_count += 1
			if(not_change_count == change_no):
				break
		else:
			not_change_count = 0

	policy =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
	plt.figure(1)
	plt.clf()
	plt.imshow(policy, interpolation='none', cmap='gray')
	plt.savefig("active_F_model/policies/source_policy%d.png" % task_num)
	return [Q, step]

def Transfer(Q, f_env, f_step = 500, epsilon = 0.3, alpha = 1.0, discount = 0.9):

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

	# gridsize = 7
	# subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ]], [[ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ]], [[ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]

	# subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ]], [[ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]

	# random.shuffle(subtasks)
	# target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

	# target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

	gridsize = 11
	subtasks = [[[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ]], [[ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ]], [[ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ]], [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ] , [ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ] , [ 8,5 ] , [ 9,6 ] , [ 8,6 ]]]

	target_task = [[ 1,7 ] , [ 0,7 ] , [ 0,6 ] , [ 0,5 ] , [ 0,4 ] , [ 0,3 ] , [ 1,3 ] , [ 2,1 ] , [ 2,2 ] , [ 2,3 ] , [ 2,4 ] , [ 3,8 ] , [ 3,7 ] , [ 3,6 ] , [ 3,5 ] , [ 3,4 ] , [ 4,4 ] , [ 0,10 ] , [ 1,10 ] , [ 2,10 ] , [ 3,10 ] , [ 4,10 ] , [ 5,10 ] , [ 5,9 ] , [ 5,8 ] , [ 5,7 ] , [ 5,6 ] , [ 5,5 ] , [ 5,4 ] , [ 6,4 ] , [ 7,4 ] , [ 8,4 ] , [ 7,10 ] , [ 8,10 ] , [ 9,10 ] , [ 10,10 ] , [ 10,9 ] , [ 10,8 ] , [ 10,7 ] , [ 10,6 ] , [ 8,5 ] , [ 9,6 ] , [ 8,6 ] , [ 8,7 ] , [ 8,8 ]]

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
			pair_features[i][j].append(1.0*x/len(tot_tasks[j]))
			pair_features[i][j].append(1)

	# print (pair_features)
	
	pair_features = np.asarray(pair_features)
	# X = []
	# y = []
	## Computing inter task tranferability matrix

	# F = np.zeros((no_tasks-1, no_tasks))
	rounds = 20

	avg_total = 0
	total_steps_F = 0
	model_no = 0
	for next_i1 in range(no_tasks-1):
		for next_j1 in range(no_tasks):
			
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

				while len(D) != num_inputs - 1:
					active_pairs.append([next_i, next_j])
					pairs.remove([next_i, next_j])
					D.append(pair_features[next_i][next_j])
					X.append([])
					X[-1].append(pair_features[next_i][next_j][0])
					X[-1].append(pair_features[next_i][next_j][1])
					# X[-1].append(pair_features[next_i][next_j][2])

					if(next_i not in source_learned): 
						s_env = Maze(gridsize, tot_tasks[next_i][:-1], tot_tasks[next_i][-1])
						[Q, step] = learn_source(next_i, s_env)
						total_steps_F += step
						source_learned.append(next_i)
						source_Q[next_i] = [Q, step]
						active_steps[next_i][next_j] += step
					else:
						[Q, step] = source_Q[next_i]
					# print ("Task pair: (%d,%d)" % (next_i, next_j))
					f_env = Maze(gridsize, tot_tasks[next_j][:-1], tot_tasks[next_j][-1])

					for r in range(rounds):
						Q2 = copy.deepcopy(Q)
						[F_reward, step] = Transfer(Q2, f_env)
						F[next_i][next_j] += F_reward
						active_steps[next_i][next_j] += step
						total_steps_F += step
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
				printCurrculum(F, no_tasks)
				X = np.asarray(X)
				y = np.asarray(y)
				y[:,0] = y[:,0]/LA.norm(y[:,0])
				# data = np.c_[X.reshape(len(X), -1), y.reshape(len(y), -1)]
				# np.random.shuffle(data)
				# X2 = data[:, :X.size//len(X)].reshape(X.shape)
				# y2 = data[:, X.size//len(X):].reshape(y.shape)
				[X_train, X_test] = np.split(X, [len(X)-48])
				[y_train, y_test] = np.split(y, [len(y)-48])
				print (len(X))
				print (len(X_train), len(X_test))
				clf = linear_model.LinearRegression()
				# clf = linear_model.Ridge (alpha = .1)
				clf.fit(X_train, y_train[:,0])
				predict_test = clf.predict(X_test)
				# print (predict_test, y_test[:,0])

				plt.figure(3)
				plt.clf()
				plt.scatter(X[:,0], y[:,0], color='red')
				predict_train = clf.predict(X_train)
				plt.scatter(X_train[:,0], predict_train, color='blue')
				plt.scatter(X_test[:,0], predict_test, color='magenta')

				predict = clf.predict(X)
				for i in range(len(X)):
					plt.plot([X[i][0], X[i][0]], [y[i][0], predict[i]], color='green')
				plt.savefig('active_F_model/model1_%d.png' % model_no)

				# plt.figure(2)
				# plt.imshow(F, interpolation='none', cmap='gray')
				# plt.savefig('active_F_model/F.png')

				for i in range(len(y_test)):
					Ts = int(y_test[i][1])
					Tf = int(y_test[i][2])
					F[Ts][Tf] = predict_test[i]

				# print (F)
				# plt.figure(2)
				# plt.imshow(F, interpolation='none', cmap='gray')
				# plt.savefig('active_F_model/active_F.png')
				printCurrculum(F, no_tasks)
				# print (total_steps_F/avg_total)
				tot_active_steps = 0
				for i in range(15):
					# print (active_pairs[i][0], active_pairs[i][1])
					# print (active_steps[active_pairs[i][0]][active_pairs[i][1]])
					tot_active_steps += active_steps[active_pairs[i][0]][active_pairs[i][1]]
				print (tot_active_steps)



