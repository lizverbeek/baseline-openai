import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

with open('rewards.csv', 'r') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=',')
	total = []
	lengths = []

	for row in csvreader:
		lengths.append(len(row))
		for i in range(len(row)):
			row[i] = float(row[i])
		total.append(np.array(row))
	m = min(lengths)

	for i in range(len(total)):
		total[i] = total[i][:m]

	average = np.mean(total, 0)
	sigma = np.std(total, 0)

	def millions(x, pos):
	    return '%1.1fM' % (x*1e-6)

	formatter = FuncFormatter(millions)

	fig, ax = plt.subplots()
	ax.xaxis.set_major_formatter(formatter)
	plt.plot(average, linewidth=0.5, color='b', alpha=0.7)
	plt.fill_between(list(range(len(average))), average-sigma, average+sigma, facecolor='b', alpha=0.3)
	plt.xlabel('number of steps')
	plt.ylabel('average return per episode')
	plt.grid(True)
	plt.show()