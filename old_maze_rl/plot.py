import csv

file=csv.reader(open('test.csv','r'))

# for row in file:
# 	print (float(row))

my_list = [[float(x) for x in row] for row in file]

print (my_list)
# with open('test.csv', 'rb') as f:

#     reader = csv.reader(f)
#     T = list(reader)
# print (T)