# test the beta*sqrt(gamma) component of the approximation in Eq. (5)
import numpy as np

iter, mean, error = np.loadtxt("stop-head.txt", skiprows=1, unpack=True)
mean[0] = -1.0/np.sqrt(np.pi)

p_res = 0.0
n_ave = 0.0
for i in range(10000):
    p_res -= mean[i]
    n_ave += (i+1)*mean[i]
    print(i+1, n_ave, n_ave + p_res*(i+1), 1.4*np.sqrt(np.log(i+1)))
