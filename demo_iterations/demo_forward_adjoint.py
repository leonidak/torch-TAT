#  This is a demo program illustrating the use for the "torch_adj_inv" function
#  for computing the application of the adjoint operator to data given on the time-space cylinder,
#  and the use of the "torch_forward" to compute the forward problem of TAT/PAT.
#
#  This script uses forward/adjoint iterations to solve the inverse problem
#  of TAT/PAT with data known on a 180 degree arch (subset of the unit circle).
#
#  The ground truth (=initial pressure) is modeled by the square 257x257 array that
#  can be read from the file "micro57.npy".
#
#  The data come from a forward computation that was done by a kWave-like algorithm (not included here)
#  with half of the data set to zero. The result saved in the file "clean_180_data.npy".
#
#  The present demo script reads the forward data from the above file, and solves the inverse
#  problem by iterating the forward/adjoint composition of operators.
#  The very first computation is done with the flags
#     lprecompute_adj = True,
#     lprecompute_forward = True.
#
#  All the concecutive calls to the forward and adjoint functions are done with these flags set to False.
#
#  To obtain the adjoint when calling "torch_adj_inv", parameter optype should be set to "adjoint".
#
#  The result of the image reconstruction can be compared to the ground truth "micro257.npy" to evaluate
#  the accuracy of the reconstruction.
#  The comparison can be done by the script "compare_res" (included). This script should
#  lie on the path found by Python....
#
#  Total computation time is reported when the script is completed.
#
#--------------------------------------------------------------------------------------

import sys
sys.path.append('../')
import numpy as np
import torch
from os import system
from random import random
from torch_TAT import torch_forward
from torch_TAT import torch_adj_inv

from time import time

#-------------------------------------------------------------------------------------------------------
def npyread(fpath):

    data = np.load(fpath)
    arr = np.zeros((2, 2))

    if type(data) != type(arr):
        print(fpath+'is not a 2D array !')
        print('Shutting down !')

    return data


#-------------------------------------- main program -----------------------------------------------------------------

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")



ndet = 360                                   # number of detectors
ntimes = 513                                 # numbers of time samples
tmax =  4.                                  # duration of measurements
raddet = 1.                                 # detectors radius
speed_of_sound = 1.                             # speed of sound

ndim =  257                                 # reconstruction grid

namedata = 'clean_180_data.npy'          # name of file with data

eps = 0.003

data = torch.tensor(np.load(namedata)).to(device)

start_time = time()
data_norm = torch.sqrt(torch.sum(data*data))
print(data_norm)


rad_corr = 1.

alpha = 1.

optype = 'adjoint'

#ndet_start = 30                              # how many detectors out of ndet exist
ndet_end = 181                              # how many detectors out of ndet exist

number_of_iter = 100

#------------------------------------ make some grids

x1 = torch.linspace(-1.,1.,ndim)
x2 = x1
xx1, xx2 = torch.meshgrid(x1,x2,indexing='ij')
radius = torch.sqrt(xx1*xx1 + xx2*xx2)



#---------------------------------- reading the model phantom -------------------------------------------



#---------------------------------------- use adjoint as a first approximation -----------------------------------


lprecompute_adj = True
lprecompute_forward = True

usmall = alpha*torch_adj_inv(device, ndet, ntimes, ndim, data, tmax, raddet, speed_of_sound, optype, lprecompute_adj, rad_corr)
usmall[radius > 0.975] = 0.
usmall[usmall < 0.] = 0.

np.save('iter0000.npy',usmall.data.cpu().numpy())
first_iter_norm = torch.sqrt(torch.sum(usmall*usmall))

lprecompute_adj = False

#                                         ------------- iterative corrections -----------------------------------

for j in range(number_of_iter):
   #start_it = time()
   #print('Iteration: ',j+1)
   stri = str(j+1).zfill(4)

   if(j > 0):
     lprecompute_forward = False

   proj = torch_forward(device,usmall,ndet,ntimes,tmax,raddet,speed_of_sound, lprecompute_forward)



   difference = data - proj
   difference[:,ndet_end:] = 0.



   ucorr = torch_adj_inv(device, ndet, ntimes, ndim, difference, tmax, raddet, speed_of_sound, optype, lprecompute_adj, rad_corr)



   uprev = usmall

   usmall = usmall + alpha*ucorr
   usmall[radius > 0.99] = 0.

   usmall[usmall < 0.] = 0.
   usmall[:129,:] = 0.

   udiff = usmall - uprev

   curr_iter_norm = torch.sqrt(torch.sum(udiff*udiff))


   if(curr_iter_norm/first_iter_norm < eps):
      break



usm = usmall.data.cpu().numpy()

end_time = time()

print('Ellapsed time :', end_time - start_time, ' sec    ',j+1,' iterations ' )

np.save('iter_final.npy',usm)

system('compare_res iter_final.npy micro257.npy')
