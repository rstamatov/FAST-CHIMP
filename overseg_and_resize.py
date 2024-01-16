import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imsave

import os
from scipy.ndimage import zoom

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

intensity_threshold = 20
watershed_tolerance = 0.1

if len (sys.argv) == 3:
    intensity_threshold = int(sys.argv[1])
    watershed_tolerance = float(sys.argv[2])

############################################################################################

def resize (image, target_shape = (64, 256, 256)):
    Nz, Ny, Nx = np.shape(image)
    Nz_out, Ny_out, Nx_out = target_shape    
    return zoom(image, (Nz_out/Nz, Ny_out/Ny, Nx_out/Nx), prefilter = False, order = 0)

############################################################################################

def change_bit_depth(image):
    return np.array(image, dtype = np.float32)

#############################################################################################

def watershed_oversegmentation(f, image, intensity_threshold, watershed_tolerance):

    if not os.path.exists("overseg"):
        os.mkdir("overseg")
        
    h_maxima, _ = ndi.label(extrema.h_maxima(image, h = watershed_tolerance))

    mask = np.zeros_like(image)
    mask[image > intensity_threshold] = 1
    labels = watershed(255-image, h_maxima, mask = mask.astype(bool))

    imsave("overseg/" + f, resize(labels))

#############################################################################################

def correction (z, img_2d):
	delta = 1
	if z > 5:
		delta = 25 / (25 - 0.192 * (z-5))
	img_2d[img_2d > 1] *= delta
	return img_2d

def correct_axial_dimness(img):
    corrected = np.zeros_like(img)
    (Nz, Ny, Nx) = np.shape(corrected)
    for z in range(Nz):
        corrected_2d = correction(z, img[z, :, :])
        corrected[z, :, :] = corrected_2d
    return corrected

#############################################################################################

input_file_location = "restored/"

from os import listdir
from os.path import isfile, join
all_files = [f for f in listdir(input_file_location) if isfile(join(input_file_location, f))]


if not os.path.exists("resized"):
    os.mkdir("resized")

if not os.path.exists("resized/test"):
    os.mkdir("resized/test")

if not os.path.exists("resized/test/images"):
    os.mkdir("resized/test/images")

for f in all_files:
    print("Processing file " + f)
    
    img = imread(input_file_location + "/" + f)    
    imsave("resized/test/images/" + f, resize(img))

    watershed_oversegmentation(f, imgcorrected, intensity_threshold, watershed_tolerance)
    


