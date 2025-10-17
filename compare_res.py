# This script will take two file names from the command line.
# (If less than two names are given as the parameters in the
# command line, the script will ask for additional input)
# It is assumed that these are filenames of numpy files representing
# two rectangular arrays of the same dimension.
# The first one is treated as an approximation and the second
# one as the ground truth. The relative L2 and L_infinity norms of the
# difference are computed and displayed.
#-----------------------------------------------------------------------


import numpy as np
import sys
import os


def npyread(fpath):
   print('Reading ',fpath)
   #print('-------------')
   data = np.load(fpath)
   arr = np.zeros((2,2))

   if(type(data) != type(arr)):
      print(fpath+'is not a 2D array !')
      print('Shutting down !')
   return data

def readd(fid):
    with open(fid,'rb') as f:
        bdata = f.read()
    fdata = np.frombuffer(bdata, dtype='f')
    if bdata[0:4] == b'ndnd':
        ncol = int(bdata[4:8])
        nrow = int(bdata[8:12])
        fdata = np.reshape(fdata,[nrow+1,ncol])
        fdata = fdata[1:,:]
    else:
        k = int(np.sqrt(np.size(fdata)))
        fdata = np.reshape(fdata,[k, k])
    return fdata

#------------------- Start main program -----------------------------



arguments = len(sys.argv) - 1


if(arguments == 2):
   fullpath =sys.argv[1]
   fullpath1=sys.argv[2]
else:
   print('This program requires two filenames to work')
   if(arguments > 2):
      quit()
   else:
      if(arguments <= 1):
         if(arguments == 0):
            fullpath = input("Please provide the first filename (i.e. approximation):")
         else:
            fullpath =sys.argv[1]

         print('File 1 is ',fullpath)
         fullpath1=input("Please provide the second filename (i.e. ground truth):")


(path_and_name,extension) = os.path.splitext(fullpath)
#(path,name) = os.path.split(path_and_name)
#
#if(len(path) > 0):
#   path = path+'\\'

(path_and_name1,extension1) = os.path.splitext(fullpath1)

if(extension == '.npy'):
   data = npyread(fullpath)
else:
   data = readd(fullpath)

if(extension1 == '.npy'):
   data1 = npyread(fullpath1)
else:
   data1 = readd(fullpath1)

if(np.shape(data1) != np.shape(data)):
    print('Data dimensions incompatible')
    quit()


(ncol,nrow) = np.shape(data)
number = ncol*nrow

data_to_use = data - data1    # this needs to be changed

datal2   = np.sum(data1*data1)
datalinf = np.max(np.abs(data1))

dmaxor = np.amax(data1)
dminor = np.amin(data1)

dmax = np.amax(data_to_use)
dmin = np.amin(data_to_use)


diffl2 = np.sum(data_to_use*data_to_use)
errl2 = np.sqrt(diffl2/datal2)
difflinf = np.max(np.abs(data_to_use))
errlinf = difflinf/datalinf

print("========================= Comparing ",path_and_name,"   with   ",path_and_name1,'=======================')
print("Max original   = ",dmaxor,"  Min original   =  ",dminor,"  L2 original   = ",np.sqrt(datal2/number))
print(" ")
print("Max difference = ",dmax,  "  Min difference =  ",dmin)
print(" ")
print("  L2 rel diff   = ",100.*np.sqrt(diffl2/datal2)," %    Linf rel diff   = ",100.*difflinf/datalinf," %")
print("-----------------------------------------------------------------------------------------------")


