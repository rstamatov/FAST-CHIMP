# FAST-CHIMP
## Overview
The FAST-CHIMP acronym stands for FAcilitated Segmentation and Tracking of CHromosomes In Mitosis Pipeline. It consists of a set of software tools which perform denoising, segmentation, registration and tracking. It starts with a raw hyperstack of a 3D+T time lapse and results in assigning each chromosome a unique integer value (its segmentation label), which is consistent over time. Please refer to the main text for detailed discussion on what imaging conditions must be used to obtain data with sufficient quality to benefit from this method. 
Briefly, we recommend acquiring super-resolution images with pixel size no more than 100 nm in XY, 200 nm in Z, and 10 seconds temporal resolution. The temporal resolution is the most critical parameter – 10 seconds is the absolute maximum, given the speed of mitotic chromosomes to allow for successful registration. Closer to 5 seconds is even more optimal and will result in much less manual corrections later.

## Installation
The following python packages are required and should be installed via pip or anaconda:

numpy scipy tifffile scikit-image networkx tensorflow csbdeep embedseg voxelmorph itk-elastix aicsimageio aicspylibczi  
Download and extract the FAST-CHIMP folder to a single location.

## Usage
The pipeline consists of eight steps. To avoid the brittleness inherent in modular programs, we designed each step independent of the others, except for the necessary input from previous steps. It is important to verify the output of each step and the availability of the input before each subsequent step to avoid unexpected behavior and to pinpoint the exact cause of potential errors. Ideally, one would look at the generated intermediate data in Fiji or another visualization software. All such intermediate data will be generated as subfolders of the “results” folder. 
Once the pipeline has been successfully run on your setup, it can be automated by modifying the batch.bat file with the correct folder locations. Running this batch will execute all steps without interruption. Still, we strongly recommend running the steps individually and verifying the output of each.
We list the commands below with brief comments and then explain each step in more detail.
1.	Data preprocessing adjusting for dimensions and pixel size (pixel sizes are in nanometers, numbers given as examples):

python TiffToStacks.py -filename "path/to/experiment.tif" -pixel_size 0.05 -pixel_size_z 0.17

2.	CARE denoising using the trained model:
python CARE_denoising.py

3.	Tubeness filter:
"/path/to/Fiji/ImageJ-win64.exe" -macro tubeness.ijm ,”/path/to/current/directory”
Alternatively, open the tubeness.ijm in ImageJ and modify the location argument.

4.	Resizing the denoised images and generating oversegmentation (the two parameters specify the watershed tolerance and intensity threshold):
python overseg_and_resize.py 0.01 0.1

5.	Segmentation using the trained Embedseg model:
python EMBEDSEG_predict.py

6.	Post-processing of segmentation labels to handle missing pieces:
python refine_embedseg.py

7.	Affine registration of each 3D image relative to the next in the time series:
python simple_elastix_register_pairs.py

8.	Tracking (propagation of the first segmentation over time):
python propagate.py start_t end_t

where start_t and end_t are the initial and final time points. Notice that by default time points start from 1000 (see below). 
After step 7, it is usually necessary to perform manual corrections on the first frame (the one to be propagated over time) before propagation, to avoid tracking the errors.

Replication on sample data
We provide 5 example images in the folder results/separate/. They are already resized and pixel-size-adjusted, so you can proceed directly with step 2 (denoising).

## Detailed explanation of the steps in FAST-CHIMP

1.	Data preprocessing 
The command for this step is

python TiffToStacks.py -filename "path/to/experiment.tif" -pixel_size <pixel size in XY>  -pixel_size_z <pixel size in Z>

This script has two goals. First, to split the input 4D hyperstack into a sequence of 3D stacks, one for each time point, which will be available in the folder results/separate (Fig. S1a). The script automatically numbers the stacks, starting with t1000.tif. For example, t1112.tif and t1113.tif are two separate 3D stacks, adjacent in time. Second, the script aims to resize the data to match the dimensions of the training data as closely as possible. This is critical if you want to use the pre-trained models for denoising, segmentation and registration. We found that these models perform reasonably well with a wide array of imaging conditions and different cell lines but only as long as the data have the same pixel size and dimensions as the training data:

Pixel size in XY: 0.05 µm; Pixel size in Z: 0.18 µm; Dimensions: 120 Z planes, 600 x 600 pixels.

An alternative to using this script is to do the resizing manually in Fiji using the commands “adjust size” to change the pixels size and “adjust canvass size” to change the size without affecting the resolution.
2.	Denoising
Run the CARE1 denoising prediction using the provided model by running the script "CARE_denoising.py":
python CARE_denoising.py
It will operate on all image stacks in the results/separate folder and the output will appear in a new folder named "restored". System-specific configurations might be necessary if you want to enable the GPU on your machine and run CARE on the GPU. A single image stack with dimensions (120 x 660 x 600) takes a few seconds using the GPU, and up to a minute using the CPU.
We recommend visual inspection of the denoising results at this point (Fig. S1b). It is possible that very different experimental conditions will make the provided denoising model inaccurate. In this case, you can train your own CARE model. Alternatively, you can try other denoising methods, such as N2N2 and N2V3, or simply use the raw images, without denoising. In the latter case, just rename the folder "separate" to "restored" and proceed.

3.	Tubeness filter
This standard filter enhances tubular structures and is especially useful for chromosomes (Fig. S1c). FAST-CHIMP uses the tubeness results only as a structural skeleton – after segmentation, all signal outside the tubeness structures is zeroed out. This results in more realistic and visually appealing segmentation masks. This step is implemented as a Fiji macro:
"/path/to/Fiji/ImageJ-win64.exe" -macro tubeness.ijm ,”/path/to/current/directory”
Alternatively, open the tubeness.ijm in ImageJ and modify the location argument.
The macro uses sigma = 0.28 µm, the empirically determined width of mitotic chromosomes in our data. You can experiment with other values. 

4.	Resizing and oversegmentation
Now the restored images will be resized to match the dimensions of the trained segmentation model. We also need to perform an oversegmentation, to help with the tracking step later. To perform both actions, run the script "overseg_and_resize.py":
python overseg_and_resize.py 0.01 0.1
The two arguments are Watershed segmentation parameters - intensity threshold and tolerance. You can vary them and inspect the results in the folder "results/overseg" which the script will create. The default values should be OK. You should look for an oversegmentation where a chromosome is roughly split into 5-10 chunks (Fig. S1d). To visualize the segmentations, load them in Fiji and apply a lookup table, e.g. "glasbey on dark". The folder results/overseg_fine will contain even finer segments, as it is generated by splitting the segmentations further.
The typical run time of this script is 10 seconds per (120 x 600 x 600) image stack.

![fig  S1](https://github.com/user-attachments/assets/6246dc22-1754-431e-abb3-8eb533d169d2)

Fig. S1 | Expected results at different processing steps. Maximum intensity projections of all Z planes. (a) Raw images, one per time point, produced in results/separate. (b) Denoised images, located in results/restored. (c) Result of the tubeness filter, will be available in results/tubeness. (d) Oversegmentation – will be in results/overseg and results/overseg_fine. (e) Output of the segmentation algorithm – in results/inference/prediction. (f) After tracking, each time point will adopt the colors of the first in the sequence, hence the different appearance from (e).  
5.	Segmentation
6. Apply the provided Embedseg4 model on the images in the "resized" folder. The model will place the segmentations in a folder "inference/prediction" (Fig. S1e):
python EMBEDSEG_predict.py
The Embedseg prediction takes ~5 seconds per image on a GPU, and up to a minute on the CPU.
6.	Refining segmentation labels
Sometimes the Embedseg prediction misses a segment completely due to insufficient probability of that segment to be classified as a mask. This issue is solved by juxtaposing the Embedseg masks to the oversegmentation masks. Each oversegmentation chunk which has no Embedseg counterpart is added as a new label to the Embedseg result. The command is as follows.
python refine_embedseg.py
It modifies the files in results/inference/predictions in place. The way to verify this step is to open a prediction along its raw (or denoised) counterpart in Fiji and overlay them, ensuring that all raw signal overlaps the segmentation. 
Registration
Pairwise registration is necessary for propagating the segmentation labels over time. Two substeps are necessary: elastix5 registration and Voxelmorph6 registration.

7.	Affine registration
Tracking of the chromosomes relies on pairwise registration – computing the displacement of each pixel form image t needed to match image t+1. Having computed these so-called deformation fields between each successive image pairs allows to take the first segmentation in the sequence and iteratively transform it over time to match each time point. 
Registration is computed using the denoised images, and the resulting deformation fields are then applied on the segmentation masks.
We use Voxelmorph for pairwise registration in a later step. It works on pairs of images which are already roughly aligned, so as a pre-processing step we perform affine registration (consisting of only translation, scale and shear) as follows:
python simple_elastix_register_pairs.py
The output will be generated in results/results_pairs. Inside, there will be a subfolder for each time point, containing the file TransofmParameters.0.txt. Please verify this before proceeding.

8.	Tracking (propagation of the first segmentation over time):
The command is
python propagate.py start_t end_t

where start_t and end_t are the initial and final time points. Note that by default, time points start from 1000. 
This script will now use the results computed in all previous steps to propagate the segmentation of start_t over time until end_t and place the output in results/propagated (Fig. S1f).
We recommend doing this step in batches, rather than using all time points. This will facilitate manual correction. For example, if the time series consists of 200 images, it is impractical to run propagation from beginning to end, e. g. From 1000 to 1200. Since segmentation and propagation are not perfect, errors will accumulate over time. It is instead a good idea to propagate 50 frames (e.g. 1000 to 1050) and inspect them visually. If the propagation is successful with only a few erroneously tracked chromosomes in the end, you can proceed with manual correction of this batch. If not, then take a smaller subset, 20 or 30 images, and correct those. Then re-start the propagation using the last corrected image as a new starting point. Note that in this case you must specify the location of the first image, otherwise it will be taken from results/inference/predictions/* and not account for the image you just corrected. So, suppose you placed the manually corrected images back into the results/propagated folder. Then to re-start the propagation, the command will be:
python propagate.py 1050 1100 results/propagated
Note the additional argument here.
