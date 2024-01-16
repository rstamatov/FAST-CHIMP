import numpy as np
from tifffile import imsave, imread
import sys
import os

###########################################################################################################################

def create_batch(start_t, end_t):


    if not os.path.exists("propagated"):
        os.mkdir("propagated")

    if not os.path.exists("temp"):
        os.mkdir("temp")
        
    with open("propagate.bat", 'w') as f:
        for t in range(start_t, end_t):

            if t == start_t:
                #f.write("\"../../elastix/transformix\" -in tubeness/resized/t" + str(t) +\
                #    ".tif -tp results_pairs/forward_" + str(t) + "/TransformParameters.0.txt -out temp \n")
                f.write("\"elastix/transformix\" -in \"inference/predictions/t" + str(t) +\
                    ".tif\" -tp results_pairs/forward_" + str(t) + "/TransformParameters.0.txt -out temp \n")
            else:
                f.write("\"elastix/transformix\" -in propagated/t" + str(t) + ".tif -tp results_pairs/forward_" +\
                        str(t) + "/TransformParameters.0.txt -out temp \n")
                
            
            f.write("propagate_using_segmentations_anaphase.py " + str(t) + "\n")



###########################################################################################################################

if __name__ == "__main__":

    start_t = int(sys.argv[1])
    end_t = int(sys.argv[2])

    create_batch(start_t, end_t)
        
