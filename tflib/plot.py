import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import cPickle as pickle
import csv

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]
def tick():
	_iter[0] += 1

def set(val):
	_iter[0] = val

def plot(name, value):
	_since_last_flush[name][_iter[0]] = value

def flush(csv_logname):
	prints = []

	for name, vals in sorted(_since_last_flush.items()):
		prints.append("{}\t{:.6f}".format(name, np.mean(vals.values())))
		_since_beginning[name].update(vals)

		x_vals = np.sort(_since_beginning[name].keys())
		y_vals = [_since_beginning[name][x] for x in x_vals]

		plt.clf()
		plt.plot(x_vals, y_vals)
		plt.xlabel('iteration')
		plt.ylabel(name)
		plt.savefig(name.replace(' ', '_')+'.png')

	print "iter {}\t{}".format(_iter[0], "\t".join(prints))
	_since_last_flush.clear()

	with open('log.pkl', 'wb') as f:
		pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)

	with open(csv_logname, 'w') as f:
		w = csv.writer(f)
		w.writerow(_since_beginning.keys())
		list_of_dicts = _since_beginning.values()
		zipped_values = [(x.values()) for x in list_of_dicts]
		w.writerows(zip(*zipped_values))