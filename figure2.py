import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import itertools

# switch to Physical Review compatible font
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb,txfonts}'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams['font.weight'] = 'roman'
matplotlib.rcParams['font.size'] = 8

iter, mean, error = np.loadtxt("stop-statistics.txt", skiprows=2, unpack=True)

fig, ax = plt.subplots(figsize=(3.5,2.05),tight_layout=True)

grid = np.logspace(0.000001, 1.0, num=200, endpoint=True, base=1e3)
model = 0.71*np.power(grid,-2.0)/np.sqrt(np.log(grid))

ax.set_xlim([1e-6,1])
ax.set_ylim([1,1e3])

ax.loglog(model, grid, marker='', color='lightgray', lw=4.5)
ax.loglog(mean, iter, marker='s', ms=2.75, mew=0, color='black', ls='')
ax.set_ylabel(r'$n$')
ax.set_xlabel(r'probability / ($\beta \sqrt{\gamma}$)')

plt.savefig('figure2.pdf', bbox_inches='tight', pad_inches=0.03)
