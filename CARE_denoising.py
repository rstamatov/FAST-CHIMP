#from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
#import matplotlib.pyplot as plt
from tifffile import imread, imwrite
#from csbdeep.utils import axes_dict, plot_some, plot_history
#from csbdeep.utils.tf import limit_gpu_memory
#from csbdeep.io import load_training_data

import os

import tensorflow as tf
#print(tf.test.is_built_with_cuda())

from csbdeep.models import Config, CARE
from scipy.ndimage import zoom



###########################################################################
#    Parse the arguments
###########################################################################

import sys


############################################################################################

def change_bit_depth(image):
    return np.array(image, dtype = np.float32)

#############################################################################################

axes = 'ZYX'
# Prediction
best_model_chromosomes = CARE(config = None, name='denoising models/my_model')

input_file_location = "results/separate/"
output_file_location = "results/restored"

if not os.path.exists(output_file_location):
    os.mkdir(output_file_location)

from os import listdir
from os.path import isfile, join
all_files = [f for f in listdir(input_file_location) if isfile(join(input_file_location, f))]

for f in all_files:
    print(f)
    x = imread(input_file_location + "/" + f)
    restored = best_model_chromosomes.predict(x, axes, n_tiles = (1, 4, 4))
    restored = np.array(restored, dtype = np.int16)

    # Normalize
    restored = restored / np.max(restored)
    restored *= 255

    t = int(f[-8:-4])

    imwrite(output_file_location + "/" + f, change_bit_depth(restored))
    #np.savez_compressed(output_file_location + "/" + f, restored)


