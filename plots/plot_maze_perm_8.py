import matplotlib.pyplot as plt
import csv
with open('verify.csv', 'rb') as f:
	reader = csv.reader(f)
	data = list(reader)

print (len(data), len(data[1]))