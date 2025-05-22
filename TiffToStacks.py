# pip install czifile
import czifile
import numpy as np
from tifffile import imwrite, TiffFile
import os
from scipy.ndimage import zoom

###########################################################################
# Resize to match the trained model size: 120 x 600 x 600 with pixel size 0.05 um in XY and 0.18 in Z
###########################################################################

def crop_to_target_size(img, pixel_size_xy, pixel_size_z):

    # Get the input image dimensions
    Nz, Ny, Nx = np.shape(img)
    
    # Calculate target dimensions
    target_xy = int((0.05 / pixel_size_xy) * 600)
    target_z = int((0.18 / pixel_size_z) * 120)

    # Crop at the center using this size
    margin_xy = int((Nx - target_xy) / 2)
    margin_z = int((Nz - target_z) / 2)

    if margin_xy > 0:
        cropped = img [:, margin_xy : -margin_xy, margin_xy : -margin_xy]
    else:
        margin_xy = -margin_xy
        cropped = np.pad(img, ((0, 0), (margin_xy, margin_xy), (margin_xy, margin_xy)))

    if margin_z > 0:
        cropped = cropped [margin_z : -margin_z, :, :]
    else:
        margin_z = -margin_z
        cropped = np.pad(cropped, ((margin_z, margin_z), (0, 0), (0, 0)))
        
    Nz, Ny, Nx = np.shape(cropped)

    cropped = zoom(cropped, (120/Nz, 600/Ny, 600/Nx))
    return cropped

###########################################################################
#    Parse the arguments
###########################################################################

import sys

# Initialize the dictionary to hold our parsed arguments
parsed_args = {
    "filename": [],
    "pixel_size": None,
    "pixel_size_z": None
}

# Iterating over the sys.argv list, skipping the script name
i = 1  # Start from 1 to skip the script name
while i < len(sys.argv):
    if sys.argv[i] == '-filename':
        i += 1  # Move to the next argument, which should be a filename
        while i < len(sys.argv) and not sys.argv[i].startswith('-'):
            # Accumulate filenames until we reach another argument
            parsed_args["filename"].append(sys.argv[i])
            i += 1
    elif sys.argv[i] == '-pixel_size':
        i += 1  # Move to the next argument, which should be the pixel_size
        if i < len(sys.argv) and not sys.argv[i].startswith('-'):
            # Assuming pixel_size is a single value
            parsed_args["pixel_size"] = float(sys.argv[i])
            i += 1
    elif sys.argv[i] == '-pixel_size_z':
        i += 1  # Move to the next argument, which should be the pixel_size
        if i < len(sys.argv) and not sys.argv[i].startswith('-'):
            # Assuming pixel_size is a single value
            parsed_args["pixel_size_z"] = float(sys.argv[i])
            i += 1
    else:
        print(f"Unknown argument: {argv[i]}")
        sys.exit(1)

if not os.path.exists("results"):
    os.mkdir("results")
if not os.path.exists("results/separate"):
    os.mkdir("results/separate")
    
###########################################################################
#    Read the czi metadata and cut into slices
###########################################################################

previous_end = 0
for filename in parsed_args["filename"]:

    # Determine file extension
    file_ext = os.path.splitext(filename)[-1].lower()

    if file_ext == ".czi":

        with czifile.CziFile(filename) as czi:
            # Read the image data
            img = czi.asarray()
            
            # Assuming time dimension is the first dimension
            print (img.shape)
            numT, numC, numZ, height, width, _ = img.shape

            for current_t in range(0, numT):

                print (current_t)
                current = np.squeeze(img[current_t, :, :, :, :, :])
                current = crop_to_target_size(current, parsed_args["pixel_size"], parsed_args["pixel_size_z"])
                imwrite("results/separate/t" + str(1000 + previous_end + current_t) + ".tif", current)

    elif file_ext in [".tif", ".tiff"]:
        with TiffFile(filename) as tif:
            # Read all images in the tif file
            images = tif.asarray()
            
            # Assuming time dimension is the first dimension for series
            numT = images.shape[0]

            for current_t in range(numT):
                print (current_t)
                current = images[current_t]
                current = crop_to_target_size(current, parsed_args["pixel_size"], parsed_args["pixel_size_z"])
                imwrite("results/separate/t" + str(1000 + previous_end + current_t) + ".tif", current)
    
    else:
        raise ValueError("Unsupported file format: Use .czi or .tif/.tiff files.")
        

    previous_end += numT


  
