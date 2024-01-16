# FAST-CHIMP
Facilitated Segmentation and Tracking of Chromosomes in Mitosis Pipeline

## Data arrangement
1. Please save separate timepoints as separate images in a folder, named "separate" and name them according to the time point, starting with t1000.tif. For example, t1112.tif and t1113.tif are two separate 3D stacks, adjacent in time.
2. Make sure the images are 30 - 40 micrometers across XY (ideal 25) and 20-25 micrometers in Z (ideal 21). If the dimensions are much bigger, please crop accordingly. If much smaller, you can pad with zeros, using Fiji's function "Adjust canvass size".
3. Adjust the size of the images, so that the pixel size in XY is close to 50 nm, and the pixel size in Z is close to 180 nm. These do not need to be exact but should be as close as possible. Size adjustment can be done in Fiji, using interpolation (Image --> Adjust --> Size).

## Denoising
4. Run the CARE denoising prediction using the provided model by running the scipt "CARE_denoising.py". The results will appear in a new folder named "restored".
#### Note: 
We recommend visual inspection of the denoising results at this point. It is possible that very different experimental conditions will make the provided denoising model inaccurate. In this case, you can train your own CARE model. Alternatively, you can try other denoising methods, such as N2N and N2V, or simply use the raw images, without denoising. In the latter case, just rename the folder "separate" to "restored" and proceed.

## Resizing
5. Now resize the restored images to shape (64, 256, 256), i.e. 256 pixels in XY and 64 planes in Z, again using Fiji's adjust size function. This is necessary for robust application of the segmentation model. Add these to a folder "resized".

## Segmentation
6. Apply the provided Embedseg model on the images in the "resized" folder. The model will place the segmentations in a folder "inference/prediction".

## Registration
7. Pairwise registration is necessary for propagating the segmentation labels over time. Two substeps are necessary: elastix registration and Voxelmorph registration.

#### Elastix registration
8. Run the script register_all.bat which invokes separate elastix commands for each pair of images. The results will be automatically saved in a folder called "results_pairs".

#### Voxelmorph registration
No action is necessary here, the Voxelmorph model will be used at the next step - the tracking procedure.

## Tracking
9. Run the batch file propagate_all.bat. It uses the python script.

## Visualize results and perform manual corrections

