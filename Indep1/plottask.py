import numpy as np
import matplotlib.pyplot as plt
from maze import Maze
import itertools
from copy import deepcopy


subtasks = [[[ 4,0 ] , [ 4,1 ], [ 4,2 ] , [ 4,3 ] , [ 4,4 ]], [[ 4,0 ] , [ 4,1 ] , [ 5,2 ] , [ 6,2] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]], [[ 4,0 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]], [[ 0, 0 ],  [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]]

target_task = [[ 0,0 ] , [ 1,1 ] , [ 1,2 ] , [ 1,3 ] , [ 1,4 ] , [ 1,5 ] , [ 1,0 ] , [ 2,0 ] , [ 3,0 ] , [ 4,0 ] , [ 6,6 ] , [ 6,5 ] , [ 6,4 ] , [ 6,3 ] , [ 6,2 ] , [ 4,1 ] , [ 5,2 ] , [ 4,2 ] , [ 4,3 ] , [ 4,4 ]]

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
		env = Maze(grid_size, task[:-1], task[-1])
		env.draw("task/%d/"%perm, task_num)

env = Maze(grid_size, target_task[:-1], target_task[-1])
env.draw("task/target")