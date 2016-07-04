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
x = [i for i in range(1, 26)]
# x.append('base')
y = [5370, 5498, 5257, 5434, 5630, 5875, 7509, 7392, 6987, 7527, 7897, 7360, 4767, 5002, 4993, 5287, 5394, 5258, 8667, 9500, 8846, 8925, 8580, 9401, 9508]
width = 1/3.0
barlist = plt.bar(x, y, width, color = "blue")
red_patch = mpatches.Patch(color='red', label='Baseline')
# green_patch = mpatches.Patch(color='yellow', label='Greedy Selection')
green_patch = mpatches.Patch(color='yellow', label='Inter-task transferability')
# plt.legend(handles=[red_patch])
# plt.legend(handles=[green_patch])
plt.legend(handles=[red_patch, green_patch])
plt.ylim([0,12000])
plt.xlabel('curriculums')
plt.ylabel('Number of steps')
barlist[24].set_color('r')
# barlist[12].set_color('y')
barlist[1].set_color('y')
barlist[0].set_color('y')
barlist[3].set_color('y')
barlist[6].set_color('y')
barlist[12].set_color('y')
barlist[13].set_color('y')
barlist[14].set_color('y')
barlist[15].set_color('y')
barlist[17].set_color('y')
plt.savefig('verify2.png')