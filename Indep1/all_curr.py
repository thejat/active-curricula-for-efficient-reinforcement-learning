
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
from maze import Maze
import itertools
from copy import deepcopy
import csv

FLAG_policy = False
# In[5]:

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


# In[21]:

def plot_policy(Q, num):
	plt.figure(1)
	plt.clf()
	grid_size = len(Q)
	plot =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
	plt.imshow(plot, interpolation='none', cmap='gray')
	if num == 0:
		plt.savefig("policies/final_policy.png")
	else:
		plt.savefig("policies/policy_%d.png" % (num))


# In[22]:

def learnTask(num, task, Q, epsilon = 0.3, alpha = 0.6, discount = 0.9):

    grid_size = Q.shape[0]
    env = Maze(grid_size, task[:-1], task[-1])
    if num != -1:
        env.draw("task/", num)
    # print grid_size
    num_actions = env.num_actions

    ## Learning source task

    tot_step = 0 # to count total no. of steps
    episode = 0 # to count total no. of episodes
    not_change_count = 0 # to check if Q function is changed or not
    change_no = 5 # required number of episodes for which Q function should be unchanged before stopping

    while (True):
        env.reset()
        game_over = False
        max_step = 100  # max number of steps for an episode, after max_iter steps, the episode ends
        step = 0
        episode += 1
        Q2 = deepcopy(Q)
        # print "hi"
        while not (game_over or step > max_step):
            step += 1
            curr_state = env.state()
            if np.random.rand() <= epsilon:  # epsilon-greedy policy
                action = np.random.randint(0, num_actions)
            else:
                if(np.amax(Q[curr_state[0]][curr_state[1]]) == np.amin(Q[curr_state[0]][curr_state[1]])):
                    action = -1
                    # if Q[] function is unable to select action, then no action taken
                else:
                    action = np.argmax(Q[curr_state[0]][curr_state[1]])
                    # best action from Q table
            next_state, reward, game_over = env.act(action)
            # Q-learning update
            Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + alpha*(reward + discount*np.amax(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])
        tot_step += step
        if (step > max_step):
            not_change_count = 0
        elif not change(Q, Q2, env):
            not_change_count += 1
            if(not_change_count == change_no):
                break
        else:
            not_change_count = 0

        if FLAG_policy == True:
            if (episode-1)%50 == 0:
                plot_policy(Q, episode)
    # print("Total no. of episodes: %d" %episode)
    # print("Total no. of steps: %d" %tot_step)
    return [Q, tot_step]


# In[23]:

subtasks = [[[ 4,0 ] , [ 4,1 ], [ 4,2 ] , [ 4,3 ] , [ 4,4 ]], [[ 4,0 ] , [ 4,1 ] , [ 5,2 ] , [ 6,2] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]], [[ 4,0 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]], [[ 0, 0 ],  [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]]

target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]


# In[24]:

def fact(n):
    if(n == 1):
        return 1
    else:
        return n*fact(n-1)


# In[25]:

no_tasks = len(subtasks)+1
grid_size = 7
print (no_tasks)
all_steps = np.zeros(fact(no_tasks-1)+1)
print all_steps.size
Rounds = 30

STEP = []
# In[ ]:

for Round in range(Rounds):
    print("Round no. %d" % (Round+1))
    perm = 0
    ind = 0
    for T in itertools.permutations(subtasks):
        perm += 1
        Q = np.zeros((grid_size, grid_size, 4))
        tot_step = 0
        task_num = 1
        for task in T:
#             if(task_num == 1):
#                 [Q, step] = learntask(perm, task_num, task, Q, grid_size, alpha = 0.2)
#                 # print (step)
#             else:
            [Q, step] = learnTask(-1, task, Q, alpha = 0.6)
            tot_step += step
            task_num += 1 
        # print(tot_step)
        [Q, step] = learnTask(ind+1, target_task, Q, alpha = 0.6)
        # print(tot_step)
        all_steps[ind] += tot_step
        ind += 1
        # print(perm, tot_step)
        # plot_policy(perm, Q)

    # BASELINE
    print ('baseline')
    Q = np.zeros((grid_size, grid_size, 4))
    [Q, step] = learnTask(0, target_task, Q, alpha = 0.9)
    all_steps[ind] += step
    print (all_steps/(Round+1))
    STEP.append(all_steps/(Round+1))
all_steps = (all_steps/Rounds)
for i in all_steps:
    print i,","
STEP = np.asarray(STEP)
np.savez("base2", STEP=STEP)
#     with open("verify_new_subtasks.csv", "w") as fp:
#         wr = csv.writer(fp)
#         for i in all_steps:
#             wr.writerow([i/(Round+1)])
    # plot_policy(0, Q)


# In[ ]:



