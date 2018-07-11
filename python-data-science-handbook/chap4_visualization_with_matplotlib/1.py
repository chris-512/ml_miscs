#!/usr/bin/env python3

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

'''
we need
 - a figure
 - an axes
 
In Matplotlib, the figure can be thought of as a single 
container that contains all the objects representing 
axes, graphics, text, and labels.

The axes is what we see above: a bounding box with tickets and labels,
which will eventually contain the plot elements that make up 
our visualization.
'''

fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 1000)
#ax.plot(x, np.sin(x))
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

plt.show()

ax = plt.axes()
ax.plot(x, np.sin(x))
ax.set(xlim=(0, 10), ylim=(-2, 2), xlabel='x', ylabel='sin(x)',
       title='A Simple Plot')
plt.show()