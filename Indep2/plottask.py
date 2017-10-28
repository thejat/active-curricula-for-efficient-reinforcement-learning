import numpy as np
import matplotlib.pyplot as plt
from grid_world import Grid
import itertools
from copy import deepcopy


block_states = [[3,0], [3,2], [3,3], [3,4], [3,5], [1,4], [2,4], [4,3], [5,3], [6,3]]

goal = [6,2]
sub_start_states = [[4,1], [1,2], [0,5], [4,6]]
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

def fact(n):
    if(n == 1):
        return 1
    else:
        return n*fact(n-1)

no_tasks = len(subtasks)+1
grid_size = 7
perm  = 0
for T in itertools.permutations(subtasks):
	perm += 1
	task_num = 0
	for task in T:
		task_num += 1
		env = Grid(grid_size, task[0], task[1], task[2])
		env.draw("task/%d/"%perm, task_num)
env = Grid(grid_size, target_task[0],target_task[1], target_task[2])
env.draw("task/target")