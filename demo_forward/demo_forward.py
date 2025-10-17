#  This is a demo program illustrating the use for the "torch_forward" function.
#  The initial pressure is modeled by the square 257x257 array that is read from the
#  file "micro_two_257.npy".
#  The forward problem computed first three times (the results are identical)
#  On the first run the flag "lprecompute_forward" is set to True,
#  all the other runs are with lprecompute_forward = False.
#  After fthe first three runs, the computation is repeated a hundred times for
#  timing purposes.
#
#  A more accurate forward computation (with a kWave-like algorithm, not included here)
#  produces the file "accurate_full_data.npy". The results of the present computation
#  are compared to the accurate result to evaluate the accuracy of the algorithm.
#  The comparison is done by the script "compare_res" (included). The script should
#  be placed on the path found by Python....
#--------------------------------------------------------------------------------------

import sys
sys.path.append('../')
import numpy as np
import torch
from os import system

from torch_TAT import torch_forward
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
ntimes = 513


                                 # numbers of time samples
tmax =  4.                                  # duration of measurements
raddet = 1.                                 # detectors radius
speed_of_sound = 1.                             # speed of sound




nameinit = 'micro_two_257.npy'
init_cond = torch.from_numpy(npyread(nameinit).astype('float32')).to(device)

print('------------------------------------------------------- Run one ------------------------------------------')


lprecompute_forward = True

proj = torch_forward(device,init_cond,ndet,ntimes,tmax,raddet,speed_of_sound, lprecompute_forward)

np.save('fast_forward',proj.data.cpu().numpy())


print('------------------------------------------------------- Run two ------------------------------------------')

lprecompute_forward = False

proj = torch_forward(device,init_cond,ndet,ntimes,tmax,raddet,speed_of_sound, lprecompute_forward)

np.save('fast_forward_run2',proj.data.cpu().numpy())


print('------------------------------------------------------- Run three ------------------------------------------')

lprecompute_forward = False

proj = torch_forward(device,init_cond,ndet,ntimes,tmax,raddet,speed_of_sound,lprecompute_forward)

np.save('fast_forward_run3',proj.data.cpu().numpy())



print('------------------------------------------------------- Run a lot ------------------------------------------------')


lprecompute_forward = False


start_t = time()

for j in range(100):
   if(j % 20 == 0):
      print('Try number ',j,'------------------------------------')
   proj = torch_forward(device,init_cond,ndet,ntimes,tmax,raddet,speed_of_sound,lprecompute_forward)


end_t = time()

print(' ')
print('====================================================================')
print('Ellapsed time: after hundred runs ','{:07.4f}'.format(0.01*(end_t-start_t)),' sec per one run')
print('====================================================================')
print(' ')

np.save('fast_forward_100',proj.data.cpu().numpy())

system('compare_res fast_forward_100.npy accurate_full_data.npy')



quit()
