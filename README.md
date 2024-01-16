# FAST-CHIMP
Facilitated Segmentation and Tracking of Chromosomes in Mitosis Pipeline

## Data arrangement
Please save separate timepoints as separate images in a folder, named "separate" and name them according to the time point, starting with t1000.tif. For example, t1112.tif and t1113.tif are two separate 3D stacks, adjacent in time. Adjust the size of the images, so that the pixel size in XY is close to 50 nm, and the pixel size in Z is close to 180 nm. These do not need to be exact but should be as close as possible. Size adjustment can be done in Fiji, using interpolation (Image --> Adjust --> Size).

## Denoising
Run the CARE denoising prediction using the provided model, giving it the "separate" folder as input. Please save the denoised results in a folder "restored".

## Resizing
Now resize the restored images to shape (64, 256, 256), i.e. 256 pixels in XY and 64 planes in Z, again using Fiji's adjust size function. This is necessary for robust application of the segmentation model. Add these to a folder "resized".

## Segmentation
Apply the provided Embedseg model on the images in the "resized" folder. The model will place the segmentations in a folder "inference/prediction".

## Registration
Pairwise registration is necessary for propagating the segmentation labels over time. Two substeps are necessary: elastix registration and Voxelmorph registration.

### Elastix registration
Run the script register_all.bat which invokes separate elastix commands for each pair of images. The results will be automatically saved in a folder called "results_pairs".
