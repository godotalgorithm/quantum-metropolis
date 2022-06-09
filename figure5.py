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

size1, direct1, mix1, mix_err1, mcmc1, mcmc_err1 = np.loadtxt("simulation.txt", max_rows=11, unpack=True, usecols = (1,2,3,4,5,6))
size2, direct2, mix2, mix_err2, mcmc2, mcmc_err2 = np.loadtxt("simulation.txt", skiprows=11, max_rows=11, unpack=True, usecols = (1,2,3,4,5,6))
size3, direct3, mix3, mix_err3, mcmc3, mcmc_err3 = np.loadtxt("simulation.txt", skiprows=22, max_rows=11, unpack=True, usecols = (1,2,3,4,5,6))

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8.75/2.54,4.5/2.54), tight_layout=True)

theta1 = np.arcsin(1.0/np.sqrt(direct1))
theta2 = np.arcsin(1.0/np.sqrt(direct2))
theta3 = np.arcsin(1.0/np.sqrt(direct3))
num1 = np.floor(0.25*np.pi/theta1)
num2 = np.floor(0.25*np.pi/theta2)
num3 = np.floor(0.25*np.pi/theta3)
# NOTE: 2 GQPE operations per amplitude-amplification phase-flip oracle query
direct1b = 2*num1/np.sin((2*num1+1)*theta1)**2
direct2b = 2*num2/np.sin((2*num2+1)*theta2)**2
direct3b = 2*num3/np.sin((2*num3+1)*theta3)**2

ax1.plot([1,13], [8*2,8*2**13], color='gray', lw=0.75)
ax1.plot([1,13], [0.5*np.pi*np.sqrt(8*2),0.5*np.pi*np.sqrt(8*2**13)], color='lightgray', lw=0.75)
ax1.plot(size1, direct1b, marker='v', color='lightgray', ls='', ms=2.5, label='amplification')
ax1.plot(size2, direct2b, marker='_', color='lightgray', ls='', ms=6.0)
ax1.plot(size3, direct3b, marker='^', color='lightgray', ls='', ms=2.5)
ax1.plot(size1, direct1, marker='v', color='gray', ls='', ms=2.5, label='postselection')
ax1.plot(size2, direct2, marker='_', color='gray', ls='', ms=6.0)
ax1.plot(size3, direct3, marker='^', color='gray', ls='', ms=2.5)
ax1.errorbar(size1, mcmc1, yerr=mcmc_err1, marker='', color='black', ms=2.5, ls='', elinewidth=1.25, zorder=100)
ax1.errorbar(size2, mcmc2, yerr=mcmc_err2, marker='', color='black', ms=6.0, ls='', elinewidth=1.25, zorder=100)
ax1.errorbar(size3, mcmc3, yerr=mcmc_err3, marker='', color='black', ms=2.5, ls='', elinewidth=1.25, zorder=100)
ax1.plot(size1, mcmc1, marker='v', color='black', ms=2.5, ls='', zorder=100, label='Metropolis')
ax1.plot(size2, mcmc2, marker='_', color='black', ms=6.0, ls='', zorder=100)
ax1.plot(size3, mcmc3, marker='^', color='black', ms=2.5, ls='', zorder=100)
ax1.legend(frameon=False,handletextpad=0.5,borderpad=0.0,handlelength=0.5,fontsize=7)
ax1.set_yscale('log')
ax1.set_xlabel(r'$m$')
ax1.set_ylabel(r'GQPE operations')
ax1.set_xlim([1,13])
ax1.set_ylim([np.sqrt(10),1e5])
#ax1.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base = 10.0, numticks = 8))
#ax1.yaxis.set_major_locator(ticker.MaxNLocator(2, steps=[7.11]))
ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax1.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base = 10.0, numticks = 8))
#ax1.xaxis.set_major_locator(ticker.FixedLocator([0,2e5,4e5,6e5,8e5,1e6]))
#ax1.xaxis.set_major_formatter(ticker.FixedFormatter([r"$0$",r"$2 \! \times \! 10^5$",r"$4 \! \times \! 10^5$",r"$6 \! \times \! 10^5$",r"$8 \! \times \! 10^5$",r"$10^6$"]))

ax2.plot(size3, mix3, marker='^', color='black', ms=2.5, ls='', label=r'$\theta = \pi/4$')
ax2.plot(size2, mix2, marker='_', color='black', ms=6.0, ls='', label=r'$\theta = \pi/8$')
ax2.plot(size1, mix1, marker='v', color='black', ms=2.5, ls='', label=r'$\theta = 0$')
ax2.errorbar(size3, mix3, yerr=mix_err3, marker='', color='black', ms=2.5, ls='', elinewidth=1.25, zorder=100)
ax2.errorbar(size2, mix2, yerr=mix_err2, marker='', color='black', ms=6.0, ls='', elinewidth=1.25, zorder=100)
ax2.errorbar(size1, mix1, yerr=mix_err1, marker='', color='black', ms=2.5, ls='', elinewidth=1.25, zorder=100)
ax2.legend(frameon=False,handletextpad=0.5,borderpad=0.0,handlelength=0.75,fontsize=7)
ax2.set_xlabel(r'$m$')
ax2.set_ylabel(r'mixing time')
ax2.set_xlim([1,13])
ax2.set_ylim([0,45])
#ax2.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base = 10.0, numticks = 8))
ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))

plt.savefig('figure5.pdf', bbox_inches='tight', pad_inches=0.025)
