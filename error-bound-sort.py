# Sort the error bound outputs that were misordered by OpenMP parallelization

import numpy as np

unsorted_data = np.loadtxt("error-bound-unsorted.txt")
for row in unsorted_data:
    row[3] *= -1
sorted_data1 = unsorted_data[np.argsort(unsorted_data[:,3], kind='stable')]
sorted_data2 = sorted_data1[np.argsort(sorted_data1[:,2], kind='stable')]
sorted_data3 = sorted_data2[np.argsort(sorted_data2[:,1], kind='stable')]
sorted_data4 = sorted_data3[np.argsort(sorted_data3[:,0], kind='stable')]
for row in sorted_data4:
    row[3] *= -1
np.savetxt("error-bound.txt", sorted_data4, fmt="%d %e %d %e   %e %e %e   %e %e")
