# FAST-CHIMP
Facilitated Segmentation and Tracking of Chromosomes in Mitosis Pipeline

## Data arrangement
Please save separate timepoints as separate images in a folder, named "separate" and name them according to the time point, starting with t1000.tif. For example, t1112.tif and t1113.tif are two separate 3D stacks, adjacent in time. Adjust the size of the images, so that the ipxel size in XY is close to 50 nm, and the pixel size in Z is close to 180 nm. These do not need to be exact but should be as close as possible. Size adjustment can be done in Fiji, using interpolation (Image --> Adjust --> Size).

## Denoising
Run the CARE denoising prediction using the provided model, giving it the "separate" folder as input. Please save the denoised results in a folder "restored".
