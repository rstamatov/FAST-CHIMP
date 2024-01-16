import numpy as np
from tifffile import imread, imsave
from csbdeep.models import Config, CARE
from os import listdir
from os.path import isfile, join
import os

############################################################################################

def change_bit_depth(image):
    return np.array(image, dtype = np.float32)

#############################################################################################

axes = 'ZYX'

# Load the pre-trained model
best_model_chromosomes = CARE(config = None, name='CARE denoising')

input_file_location = "separate/"
output_file_location = "restored"

# Create the output folder
if not os.path.exists(output_file_location):
    os.mkdir(output_file_location)

all_files = [f for f in listdir(input_file_location) if isfile(join(input_file_location, f))]

for f in all_files:
    
    print("denoising " + f)
    x = imread(input_file_location + "/" + f)
    restored = best_model_chromosomes.predict(x, axes, n_tiles = (1, 4, 4))
    restored = np.array(restored, dtype = np.int16)

    # Normalize
    restored = restored / np.max(restored)
    restored *= 255

    imsave(output_file_location + "/" + f, change_bit_depth(restored))


