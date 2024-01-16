from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imsave
import os

##################################################################################################################################

def reorganize_registration_results(start_t, end_t):

    from shutil import copyfile

    if not os.path.exists("affines"):
        os.mkdir("affines")
    
    for t in range(start_t, end_t):
        try:
            copyfile("results_pairs/backward_" + str(t) + "/result.0.tif", "affines/t" + str(t) + "_backward.tif")
            copyfile("results_pairs/forward_" + str(t) + "/result.0.tif", "affines/t" + str(t) + "_forward.tif")
        except Exception as e:
            print (e)

##################################################################################################################################

def tif_to_npz(start_t, end_t):
    from tifffile import imread

    if not os.path.exists("npz"):
        os.mkdir("npz")
        
    for t in range(start_t, end_t):
        print (t)

        try:
            img = imread("affines/t" + str(t) + "_forward.tif")
            np.savez("npz/t" + str(t) + "_forward.npz", img)
            img = imread("affines/t" + str(t) + "_backward.tif")
            np.savez("npz/t" + str(t) + "_backward.npz", img)
            img = imread("resized/test/images/t" + str(t) + ".tif")
            np.savez("npz/t" + str(t) + ".npz", img)
            
        except Exception as e:
            print (e)
            
##################################################################################################################################

if __name__ == "__main__":

    import sys

    start_t = int(sys.argv[1])
    end_t = int(sys.argv[2])

    reorganize_registration_results(start_t, end_t)
    tif_to_npz(start_t, end_t)

    if not os.path.exists("registration models/instance"):
        os.mkdir("registration models/instance")
                     
