import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# import plotly.plotly as py
# py.sign_in('PythonAPI', 'ubpiol2cve')
# # Learn about API authentication here: https://plot.ly/python/getting-started
# # Find your api_key here: https://plot.ly/settings/api

# y = [5370, 5498, 5257, 5434, 5630, 5875, 7509, 7392, 6987, 7527, 7897, 7360, 4767, 5002, 4993, 5287, 5394, 5258, 8667, 9500, 8846, 8925, 8580, 9401, 9508]
# N = len(y)
# x = range(N)
# width = 1/3.0
# plt.bar(x, y, width, color="blue")


# fig = plt.gcf()
# plot_url = py.plot_mpl(fig, filename='verify')
x = [i for i in range(1, 25)]
# x.append('base')
y = [13732.48	,
13713.32	,
14367.98	,
13896.2	,
13984.68	,
14096.52	,
14791.22	,
15180.88	,
14605.62	,
14323.1	,
13868.6	,
14533.92	,
14299.48	,
14206.66	,
14862.82	,
14719.58	,
14962.14	,
14669.7	,
15169.2	,
15205.22	,
15789.3	,
14591.88	,
16040.98	,
15038.28	
]
# y = [5370, 5498, 5257, 5434, 5630, 5875, 7509, 7392, 6987, 7527, 7897, 7360, 4767, 5002, 4993, 5287, 5394, 5258, 8667, 9500, 8846, 8925, 8580, 9401, 9508]
width = 1/3.0
barlist = plt.bar(x, y, width, color = "blue")
# red_patch = mpatches.Patch(color='red', label='Baseline')
# green_patch = mpatches.Patch(color='yellow', label='Greedy Selection')
green_patch = mpatches.Patch(color='yellow', label='Curriculum Selected')
# plt.legend(handles=[red_patch])
# plt.legend(handles=[green_patch])
base, = plt.plot([0, 25], [16822.66, 16822.66], color = "red", label = "Baseline")
plt.legend(handles=[base, green_patch], fontsize = 9)
plt.ylim([0,19000])
plt.xlabel('Curricula')
# plt.yscale('log')
plt.ylabel('Number of steps')
# barlist[24].set_color('r')
barlist[0].set_color('y')
# barlist[1].set_color('y')
# barlist[0].set_color('y')
# barlist[3].set_color('y')
# barlist[6].set_color('y')
# barlist[12].set_color('y')
# barlist[13].set_color('y')
# barlist[14].set_color('y')
# barlist[15].set_color('y')
# barlist[17].set_color('y')
plt.savefig('verify2.png')