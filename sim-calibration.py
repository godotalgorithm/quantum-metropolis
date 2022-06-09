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

for i in range(17):
    if i%2:
        error.append(0.0316*pow(10.0,-(i//2)))
    else:
        error.append(0.1*pow(10.0,-(i//2)))

    step, num_iter, energy, polar = np.loadtxt(f"sim-calibration{i+1}.txt", skiprows=5, unpack=True, max_rows=num_data)

    # remove beta prefactor from energy
    energy /= 3.0

    num_block = 0
    num_block_total = 0
    energy_block = 0.0
    polar_block = 0.0

    energy_sum = 0.0
    polar_sum = 0.0

    energy_ave0 = 0.0
    energy_var0 = 0.0
    polar_ave0 = 0.0
    polar_var0 = 0.0

    # block-averaging analysis of data
    for j in range(num_data):
        energy_block += energy[j]
        polar_block += polar[j]

        if j%block_size == block_size-1:
            num_block_total += 1
            energy_block /= block_size
            polar_block /= block_size

            if num_block_total > num_cut:
                num_block += 1
                energy_ave0 += energy_block
                energy_var0 += energy_block*energy_block
                polar_ave0 += polar_block
                polar_var0 += polar_block*polar_block

            energy_block = 0.0
            polar_block = 0.0

    step_block.append(block_size*num_block)
    energy_ave.append(energy_ave0/num_block)
    energy_err.append(np.sqrt((energy_var0/num_block - (energy_ave0/num_block)**2)/num_block))
    polar_ave.append(polar_ave0/num_block)
    polar_err.append(np.sqrt((polar_var0/num_block - (polar_ave0/num_block)**2)/num_block))

    print(error[i], energy_ave[i], energy_err[i], polar_ave[i], polar_err[i])
