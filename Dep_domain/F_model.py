import numpy as np
from maze import Maze 
import copy
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from numpy import linalg as LA

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

	# print ('Exceed: %d/%d' % (exceed, episode))
	policy =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
	plt.figure(1)
	plt.clf()
	plt.imshow(policy, interpolation='none', cmap='gray')
	plt.savefig("F_model/policies/source_policy%d.png" % task_num)
	return Q

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
		# if(itr > max_iter):
		# 	print ('maximum steps exceeded, starting new episode.')
	# print (tot_reward)
	return tot_reward

if __name__ == "__main__":

	gridsize = 7
	# subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ]], [[ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ]], [[ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]

	subtasks = [[[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ]], [[ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ]], [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ]]]

	# random.shuffle(subtasks)
	# target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 3,6 ] , [ 4,6 ] , [ 5,6 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

	target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]



	# Evaluating task pairs features
	sfeatures = [[16.0/6, 0, 2], [40.0/9, 1, 2], [6.0/3, 0, 1], [76.0/16, 3, 3]]
	tfeatures = [[111.0/18, 4, 3]]
	task_features = sfeatures + tfeatures
	no_features = 3
	# I have 20 task pairs, I have to select some of them for learning the regression model. which one to choose?
	# Answer -- As of now randomly choosing some pairs as training and remaining as test.

	no_tasks = len(subtasks)+1
	pair_features = [ [[] for i in range(no_tasks)] for j in range(no_tasks-1) ]

	for i in range(no_tasks-1):
		for j in range(no_tasks):
			for feat in range(no_features):
				x = (task_features[i][feat] - task_features[j][feat])/max(task_features[i][feat], 0.01)
				pair_features[i][j].append(x)
	
	pair_features = np.asarray(pair_features)
	# print (pair_features)
	X = []
	for i in range(no_tasks-1):
		for j in range(no_tasks):
			X.append([])
			X[-1].append(pair_features[i][j][0])
			X[-1].append(pair_features[i][j][1])
			X[-1].append(pair_features[i][j][2])
	X = np.asarray(X)

	## Computing inter task tranferability matrix

	tot_tasks = copy.deepcopy(subtasks)
	tot_tasks.append(target_task)
	F = np.zeros((no_tasks-1, no_tasks))
	rounds = 20
	for Ts in range(no_tasks-1):
		s_env = Maze(gridsize, tot_tasks[Ts][:-1], tot_tasks[Ts][-1])
		s_env.draw("F_model/tasks/", Ts)
		Q = learn_source(Ts, s_env)
		for Tf in range(no_tasks):
			print ("Task pair: (%d,%d)" % (Ts, Tf))
			f_env = Maze(gridsize, tot_tasks[Tf][:-1], tot_tasks[Tf][-1])
			# print Q
			for r in range(rounds):
				Q2 = copy.deepcopy(Q)
				F[Ts][Tf] += Transfer(Q2, f_env)

	F = F/LA.norm(F)
	print (F)

	y = []
	for i in range(no_tasks-1):
		for j in range(no_tasks):
			y.append(F[i][j])

	y = np.asarray(y)
	data = np.c_[X.reshape(len(X), -1), y.reshape(len(y), -1)]
	np.random.shuffle(data)
	X2 = data[:, :X.size//len(X)].reshape(X.shape)
	y2 = data[:, X.size//len(X):].reshape(y.shape)
	[X_train, X_test] = np.split(X2, [len(X2)-3])
	[y_train, y_test] = np.split(y2, [len(y2)-3])

	clf = linear_model.LinearRegression()
	# clf = linear_model.Ridge (alpha = .1)
	clf.fit(X_train, y_train)
	predict_test = clf.predict(X_test)
	print (predict_test, y_test)

	plt.figure(3)
	plt.scatter(X2[:,0], y2, color='red')
	# fig = plt.figure(3)
	# ax = fig.add_subplot(111, projection='3d')
	# ax.scatter(X2[:,0], X2[:,1], y2, color='red')
	predict_train = clf.predict(X_train)
	plt.scatter(X_train[:,0], predict_train, color='blue')
	plt.scatter(X_test[:,0], predict_test, color='magenta')

	predict = clf.predict(X2)
	for i in range(len(X2)):
		plt.plot([X2[i][0], X2[i][0]], [y2[i], predict[i]], color='green')
	plt.savefig('F_model/model.png')

	plt.figure(2)
	plt.imshow(F, interpolation='none', cmap='gray')
	plt.savefig('F_model/F.png')

