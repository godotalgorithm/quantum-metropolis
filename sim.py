import numpy as np

num_data = 1000000
block_size = 1000
num_cut = 0

error = []
step_block = []
energy_ave = []
energy_err = []
polar_ave = []
polar_err = []

for k in ['a','b','c']:
    for i in range(2,13):

        step, num_iter, energy, polar = np.loadtxt(f"sim{i}{k}.txt", skiprows=5, unpack=True, max_rows=num_data)
        energy_err0 = float(open(f"sim{i}{k}.txt").readlines()[3].split()[11])
        p_direct = float(open(f"sim{i}{k}.txt").readlines()[4].split()[5])

        num_block = 0
        num_block_total = 0
        iter_block = 0.0
        energy_block = 0.0

        iter1 = 0.0
        iter2 = 0.0
        energy1 = 0.0
        energy2 = 0.0
        energy3 = 0.0
        energy4 = 0.0

        # block-averaging analysis of data
        for j in range(num_data):
            iter_block += num_iter[j]
            energy_block += energy[j]

            if j%block_size == block_size-1:
                num_block_total += 1
                energy_block /= block_size
                iter_block /= block_size

                if num_block_total > num_cut:
                    num_block += 1
                    iter1 += iter_block
                    iter2 += iter_block**2
                    energy1 += energy_block
                    energy2 += energy_block**2
                    energy3 += energy_block**3
                    energy4 += energy_block**4

                energy_block = 0.0

        step_block = block_size*num_block
        iter1 /= num_block
        iter2 /= num_block
        energy1 /= num_block
        energy2 /= num_block
        energy3 /= num_block
        energy4 /= num_block
        iter_err = np.sqrt((iter2 - iter1**2)/num_block)
        energy_err = np.sqrt((energy2 - energy1**2)/num_block)
        energy_err2 = np.sqrt((energy4 - 4.0*energy3*energy1 + 6.0*energy2*energy1**2 - 3.0*energy1**4 - (energy2 - energy1**2)**2)/num_block)/num_block

        print(k, i, 1.0/p_direct, step_block*(energy_err/energy_err0)**2, step_block*energy_err2/(energy_err0**2), iter1, iter_err)
