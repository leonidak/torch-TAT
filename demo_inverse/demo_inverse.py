#  This is a demo program illustrating the use for the "torch_adj_inv" function
#  for the solution of the inverse problem of TAT/PAT in circular geometry.
#
#  The ground truth (=initial pressure) is modeled by the square 257x257 array that
#  can be read from the file "micro_two_257.npy".
#
#  The forward computation was done by a kWave-like algorithm (not included here)
#  with the result saved in the file "accurate_full_data.npy".
#
#  The present demo script reads the forward data from the above file, and computes
#  the inverse, first two times and then a hundred times for the timing purposes.
#  The very first computation is done with the flag lprecompute_adj = True,
#  all concecutive runs are done with lprecompute_adj = False.
#
#  To obtain the inverse, parameter optype should be set to "inverse".
#
#  The result of the image reconstruction is compared to the ground truth to evaluate
#  the accuracy of the reconstruction.
#  The comparison is done by the script "compare_res" (included). This script should
#  lie on the path found by Python....
#--------------------------------------------------------------------------------------

from sys import path
from os import system
from time import time
import numpy as np
path.append('../')

import torch

from torch_TAT import torch_adj_inv

#from tatutils import npyread

#-------------------------------------------------------------------------------------------------------
def npyread(fpath):

    data = np.load(fpath)
    arr = np.zeros((2, 2))

    if type(data) != type(arr):
        print(fpath+'is not a 2D array !')
        print('Shutting down !')

    return data




ndet = 360                                  # number of detectors

ntimes = 513                               # numbers of time samples
namedata = 'accurate_full_data.npy'          # name of file with data

tmax = 4.                            # duration of measurements
raddet = 1.                                 # detectors radius
ndim =  257                                 # reconstruction grid
speed_of_sound = 1.                             # speed of sound
rad_corr = 1.


# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


#---------------------------------- reading the time/space cylinder input data -------------------------------------------
data_read = npyread(namedata)

data_torch = torch.from_numpy(data_read.astype("float32"))
data = data_torch.to(device)

print('--- File ',namedata,'--- was read.')




print('------------------------------------------------------- Run one ------------------------------------------')

lprecompute_adj = True


optype = 'inverse'

usmall = torch_adj_inv(device, ndet, ntimes, ndim, data, tmax, raddet, speed_of_sound, optype, lprecompute_adj, rad_corr)

usmall[usmall < 0.] = 0.

np.save('rec_fast_a.npy',usmall.data.cpu().numpy())

#np.save('rec_large_a.npy',ularge.data.cpu().numpy())





print('------------------------------------------------------- Run two ------------------------------------------')

lprecompute_adj = False

optype = 'inverse'

usmall = torch_adj_inv(device, ndet, ntimes, ndim, data, tmax, raddet, speed_of_sound, optype, lprecompute_adj, rad_corr)

np.save('rec_fast_b.npy',usmall.data.cpu().numpy())



print('------------------------------------------------------- Run a lot ------------------------------')

lprecompute_adj = False
optype = 'inverse'


start_t = time()

for j in range(100):
   if(j % 20 == 0):
      print('Try number ',j,'------------------------------------')

   usmall = torch_adj_inv(device, ndet, ntimes, ndim, data, tmax, raddet, speed_of_sound, optype, lprecompute_adj, rad_corr)


end_t = time()
print('After hundred runs ','{:07.4f}'.format(0.01*(end_t-start_t)),' sec')


np.save('rec_fast_100',usmall.data.cpu().numpy())


system('compare_res rec_fast_100.npy micro_two_257.npy')
quit()

