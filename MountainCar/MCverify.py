from MountainCarDemo import *
from copy import deepcopy
import sys

def change(Q1, Q2):
	thres = 0.00001 
	for i in range(len(Q1)):
		prev_val = sum(Q1[i])
		new_val = sum(Q2[i])
		# print (prev_val, new_val)
		if(abs(prev_val - new_val) > thres):
			change = 1
			break
		else:
			change = 0
	return change

def learn( start, goal, tQ, budget = 100000):

	MC  = MountainCar(start, goal, tQ)
	maxsteps = 1000
	grafica  = False

	stop  = False

	not_change_count = 0
	change_no = 5

	tot_episodes = 0
	tot_steps = 0
	tot_reward = 0
	while(tot_steps < budget):

		Q_prev = deepcopy(MC.retQTable())
		reward, steps = MC.QLearningEpisode(maxsteps, grafica)
		sys.stdout.write ('\rEspisode: %4d Steps: %6d Reward %5d Epsilon: %0.3f' %(tot_episodes, steps, reward, MC.epsilon))
		sys.stdout.flush()
		MC.epsilon = MC.epsilon * 0.99	
		Q = deepcopy(MC.retQTable())

		tot_episodes += 1
		tot_steps += steps
		tot_reward += reward

		# if not change(Q_prev, Q):
		# 	not_change_count += 1
		# 	if(not_change_count == change_no):
		# 		break
		# else:
		# 	not_change_count = 0

	# print (Q_prev)

	print ('\nTotal epsiodes: %d' %tot_episodes)
	print ('Total steps: %d' %tot_steps)
	print ('Total Reward: %d' %tot_reward)

	return Q

if __name__ == "__main__":

	# position -1.5 to 0.55
	# velocity -0.7 tp 0.7

	# subtasks + target task
	tasks = [[0.2, 0.0], [-0.5, 0.5], [-0.5, 0.0]]

	# learning task
	start = tasks[0]
	goal = 0.45
	tQ = [[0.0 for i in range(3)] for i in range(66)]
	Q = learn(start, goal, tQ)

	# target task with transfer
	tQ = Q
	start = tasks[2]
	goal = 0.45
	Qf = learn(start, goal, tQ)

	# target task alone
	tQ = [[0.0 for i in range(3)] for i in range(66)]
	Q = learn(start, goal, tQ)