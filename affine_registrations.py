import numpy as np
import matplotlib.pyplot as plt
from tifffile import imsave
import os

def register_pairs(elastix_location, start_t, end_t):

    """ Pairwise elastix registration. Compute the transformation necessary
        to match image (t+1) to image (t). The result is a batch file which will
        call elastix with the appropriate parameters. """

    # Create the results folders
    if not os.path.exists("results_pairs"):
        os.mkdir("results_pairs")

    # Create the batch file which will hold sequential elastix commands for each pair
    f = open("register_pairs.bat", "w+")

    for t in range(start_t, end_t + 1):

        # Create the results folders
        if not os.path.exists("results_pairs/backward_" + str(t)):
            os.mkdir("results_pairs/backward_" + str(t))
        if not os.path.exists("results_pairs/forward_" + str(t)):
            os.mkdir("results_pairs/forward_" + str(t))

        """ Backward registration """            
        # Create the command
        f.write("\"" + elastix_location + "/elastix\" ")
        f.write("-f \"" + "resized/test/images/" + "t" + str(t-1).zfill(4) + ".tif\" ")

        # Moving image (to be deformed to match the fixed): the next timepoint
        f.write("-m \"" + "resized/test/images/" + "t" + str(t).zfill(4) + ".tif\" ")

        # Output
        f.write("-out \"" + "results_pairs/backward_" + str(t) + "\" ")

        # The parameter files and the mask should be placed in this folder manually
        f.write("-p par_affine.txt \n")

        """ Forward registration """
        # Create the command
        f.write("\"" + elastix_location + "/elastix\" ")
        f.write("-m \"" + "resized/test/images/" + "t" + str(t-1).zfill(4) + ".tif\" ")

        # Moving image (to be deformed to match the fixed): the next timepoint
        f.write("-f \"" + "resized/test/images/" + "t" + str(t).zfill(4) + ".tif\" ")

        # Output
        f.write("-out \"" + "results_pairs/forward_" + str(t-1) + "\" ")

        # The parameter files and the mask should be placed in this folder manually
        f.write("-p par_affine.txt \n")

    f.close()
    
##################################################################################################################################

if __name__ == "__main__":

    import sys

    start_t = int(sys.argv[1])
    end_t = int(sys.argv[2])
    
    register_pairs("elastix", start_t, end_t)
                     
