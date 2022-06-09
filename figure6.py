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

num_data = 1000000
step, num_iter, energy, polar = np.loadtxt("simulation.txt", skiprows=5, unpack=True, max_rows=num_data)

#fig, [ax1, ax2, ax3] = plt.subplots(3, 1, gridspec_kw={'height_ratios': [2, 2, 1]}, figsize=(8.75/2.54,6.95/2.54), constrained_layout=True)
fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(8.75/2.54,4.6/2.54), constrained_layout=True)

# remove beta prefactor from energy
energy *= 0.5

num_block = 0
energy_block = 0.0
polar_block = 0.0

energy_sum = 0.0
polar_sum = 0.0

energy_ave0 = 0.0
energy_var0 = 0.0
polar_ave0 = 0.0
polar_var0 = 0.0

step_block = []
energy_ave = []
energy_err = []
polar_ave = []
polar_err = []

block_size = 1000
sub_sample = 25

# block averaging & subsampling of data
for i in range(num_data):
    energy_block += energy[i]
    polar_block += polar[i]

    if i%block_size == block_size-1:
        num_block += 1
        energy_block /= block_size
        polar_block /= block_size

        energy_ave0 += energy_block
        energy_var0 += energy_block*energy_block
        polar_ave0 += polar_block
        polar_var0 += polar_block*polar_block

        energy_block = 0.0
        polar_block = 0.0

        if num_block%sub_sample == 0:
            step_block.append(block_size*num_block)
            energy_ave.append(energy_ave0/num_block)
            energy_err.append(np.sqrt((energy_var0/num_block - (energy_ave0/num_block)**2)/num_block))
            polar_ave.append(polar_ave0/num_block)
            polar_err.append(np.sqrt((polar_var0/num_block - (polar_ave0/num_block)**2)/num_block))

# exact reference values
energy0 = -20.08850 * 0.5
polar0 = -0.5856016

ax1.errorbar(step_block, polar_ave, yerr=polar_err, marker='o', color='black', ms=2.5, ls='', elinewidth=1.25, zorder=100)
ax1.plot([0,num_data+block_size*sub_sample],[polar0,polar0], marker='', color='darkgray')
ax1.set_ylabel(r'polarization')
ax1.set_xlim([0,num_data+block_size*sub_sample])
ax1.xaxis.set_major_formatter(ticker.NullFormatter())
ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))

ax2.errorbar(step_block, energy_ave, yerr=energy_err, marker='o', color='black', ms=2.5, ls='', elinewidth=1.25, zorder=100)
ax2.plot([0,num_data+block_size*sub_sample],[energy0,energy0], marker='', color='darkgray')
ax2.set_xlabel(r'number of Metropolis operations')
ax2.set_ylabel(r'energy')
ax2.set_xlim([0,num_data+block_size*sub_sample])
ax2.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
ax2.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
ax2.xaxis.set_major_locator(ticker.FixedLocator([0,2e5,4e5,6e5,8e5,1e6]))
ax2.xaxis.set_major_formatter(ticker.FixedFormatter([r"$0$",r"$2 \! \times \! 10^5$",r"$4 \! \times \! 10^5$",r"$6 \! \times \! 10^5$",r"$8 \! \times \! 10^5$",r"$10^6$"]))

#ax3.hist(num_iter, bins=np.logspace(start=np.log10(1), stop=np.log10(1e7), num=36), log=True, color='darkgray', edgecolor='black', lw=1)
#ax3.set_xscale('log')
#ax3.set_xlim([1,1e7])
#ax3.set_ylim([1,1e6])
#ax3.set_xlabel(r'number of GQPE operations per Metropolis operation')
#ax3.set_ylabel('count')
#ax3.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=10))
#ax3.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=10))
#ax3.xaxis.set_minor_formatter(plt.NullFormatter())
#ax3.yaxis.set_minor_formatter(plt.NullFormatter())

plt.savefig('figure5.pdf', bbox_inches='tight', pad_inches=0.01)
