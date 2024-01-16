from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imsave
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
import os
from scipy.ndimage import zoom
from skimage.filters import sato

# for watershed
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import extrema


###########################################################################
#    Parse the arguments
###########################################################################

import sys

intensity_threshold = 50
watershed_tolerance = 0.1

if len (sys.argv) == 3:
    intensity_threshold = int(sys.argv[1])
    watershed_tolerance = float(sys.argv[2])

############################################################################################

def tubeness(image):
    return sato(image, sigmas = [4], black_ridges = False)

def resize (image, target_shape = (64, 256, 256)):
    Nz, Ny, Nx = np.shape(image)
    Nz_out, Ny_out, Nx_out = target_shape    
    return zoom(image, (Nz_out/Nz, Ny_out/Ny, Nx_out/Nx), prefilter = False, order = 0)

############################################################################################

def change_bit_depth(image):
    return np.array(image, dtype = np.float32)

#############################################################################################

def watershed_oversegmentation(t, image, intensity_threshold, watershed_tolerance):

    if not os.path.exists("overseg"):
        os.mkdir("overseg")
        
    h_maxima, _ = ndi.label(extrema.h_maxima(image, h = watershed_tolerance))

    mask = np.zeros_like(image)
    mask[image > intensity_threshold] = 1
    labels = watershed(255-image, h_maxima, mask = mask.astype(bool))

    imsave("overseg/t" + str(t) + ".tif", resize(labels))

#############################################################################################

def correction (z, img_2d):
	delta = 1
	if z > 5:
		delta = 25 / (25 - 0.36 * (z-5))
	img_2d[img_2d > 1] *= delta
	return img_2d

def correct_axial_dimness(img):
    corrected = np.zeros_like(img)
    for z in range(64):
        corrected_2d = correction(z, img[z, :, :])
        corrected[z, :, :] = corrected_2d
    return corrected

#############################################################################################

input_file_location = "restored/"

from os import listdir
from os.path import isfile, join
all_files = [f for f in listdir(input_file_location) if isfile(join(input_file_location, f))]

if not os.path.exists("tubeness"):
    os.mkdir("tubeness")

for f in all_files:
    print(f)
    t = int(f[-8:-4])
    
    img = imread(input_file_location + "/" + f)
    tube = tubeness(img)
    imsave("tubeness/t" + str(t) + ".tif", tube)
    
    corrected = correct_axial_dimness(tube)


    
    watershed_oversegmentation(t, corrected, intensity_threshold, watershed_tolerance)


