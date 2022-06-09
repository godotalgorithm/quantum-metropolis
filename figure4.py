import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# switch to Physical Review compatible font
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb,txfonts}'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams['font.weight'] = 'roman'
matplotlib.rcParams['font.size'] = 8

error, energy_ave, energy_err, polar_ave, polar_err = np.loadtxt("sim-calibration.txt", unpack=True)

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8.75/2.54,4.0/2.54), tight_layout=True)

# exact reference values
energy0 = -21.36465 / 3.0
polar0 = 0.6724238

energy_min = -7.12401
energy_max = -7.10

polar_min = 0.670
polar_max = 0.674

error_min = 5.6e-10
error_max = 1.8e-6

ax1.errorbar(error, energy_ave, yerr=energy_err, marker='_', color='black', ms=3, ls='', elinewidth=1.25, zorder=100)
ax1.plot([error_min,error_max],[energy0,energy0], marker='', color='lightgray', lw=4)
ax1.set_xscale('log')
ax1.set_xlabel(r'$\epsilon$')
ax1.set_ylabel(r'energy')
ax1.set_xlim([error_min,error_max])
ax1.set_ylim([energy_min,energy_max])
ax1.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base = 10.0, numticks = 8))
#ax1.yaxis.set_major_locator(ticker.MaxNLocator(2, steps=[7.11]))
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
#ax1.xaxis.set_major_locator(ticker.FixedLocator([0,2e5,4e5,6e5,8e5,1e6]))
#ax1.xaxis.set_major_formatter(ticker.FixedFormatter([r"$0$",r"$2 \! \times \! 10^5$",r"$4 \! \times \! 10^5$",r"$6 \! \times \! 10^5$",r"$8 \! \times \! 10^5$",r"$10^6$"]))

ax2.errorbar(error, polar_ave, yerr=polar_err, marker='_', color='black', ms=3, ls='', elinewidth=1.25, zorder=100)
ax2.plot([error_min,error_max],[polar0,polar0], marker='', color='lightgray', lw=4)
ax2.set_xscale('log')
ax2.set_xlabel(r'$\epsilon$')
ax2.set_ylabel(r'correlation')
ax2.set_xlim([error_min,error_max])
ax2.set_ylim([polar_min,polar_max])
ax2.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base = 10.0, numticks = 8))
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))

plt.savefig('figure4.pdf', bbox_inches='tight', pad_inches=0.01)
