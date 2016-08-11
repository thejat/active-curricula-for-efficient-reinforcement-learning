from MountainCarDemo import *
from copy import deepcopy

def change(Q1, Q2):
	thres = 0.0 
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

if __name__ == "__main__":
	
	start = [-0.5, 0.0]
	goal = 0.45
	tQ = [[0.0 for i in range(3)] for i in range(66)]
	MC  = MountainCar(start, goal, tQ)
	maxsteps = 1000
	grafica  = False

	stop  = False

	not_change_count = 0
	change_no = 5

	tot_episodes = 0
	tot_steps = 0
	while(not stop):

		Q_prev = deepcopy(MC.retQTable())
		total_reward, steps = MC.SARSAEpisode(maxsteps, grafica)
		print ('Espisode: ',tot_episodes,'  Steps:',steps,'  Reward:',str(total_reward),' epsilon: ',str(MC.epsilon))
		MC.epsilon = MC.epsilon * 0.99	
		Q = deepcopy(MC.retQTable())

		tot_episodes += 1
		tot_steps += steps

		if not change(Q_prev, Q):
			not_change_count += 1
			if(not_change_count == change_no):
				break
		else:
			not_change_count = 0

	print ('Total epsiodes: %d' %tot_episodes)
	print ('Total steps: %d' %tot_steps)


