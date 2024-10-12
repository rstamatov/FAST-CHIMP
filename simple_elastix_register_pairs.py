# pip install itk-elastix
import itk
import os
from tifffile import imread
import numpy as np

def register_command(moving_image_filename, fixed_image_filename, output_directory):
    # Read the fixed and moving images
    moving_image = itk.imread(moving_image_filename, itk.F)
    fixed_image = itk.imread(fixed_image_filename, itk.F)
    
    # Register images using elastix
    parameter_object = itk.ParameterObject.New()
    #parameter_map = parameter_object.GetDefaultParameterMap('affine')
    parameter_object.AddParameterFile('par_affine.txt')

    # Perform registration
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object = parameter_object,
        output_directory = output_directory)

    # Save the result
    return result_image

###################################################################################################################

def register_pairs():
    
    # Create the results folders
    if not os.path.exists("results/results_pairs"):
        os.mkdir("results/results_pairs")
    if not os.path.exists("results/npz"):
        os.mkdir("results/npz")
    if not os.path.exists("temp"):
        os.mkdir("temp")

    files = list(os.listdir("results/resized/test/images"))

    for img_filename in files[:-1]:

        t = int(img_filename[-8:-4])
        
        print (t)
        img = imread("results/resized/test/images/" + img_filename)
        np.savez("results/npz/t" + str(t) + ".npz", img)

        # Create the results folders
        if not os.path.exists("results/results_pairs/backward_" + str(t+1)):
            os.mkdir("results/results_pairs/backward_" + str(t+1))
        if not os.path.exists("results/results_pairs/forward_" + str(t)):
            os.mkdir("results/results_pairs/forward_" + str(t))

       
        # Backward registration
        moving_image_filename = "results/resized/test/images/t"+ str(t+1) + ".tif"
        fixed_image_filename = "results/resized/test/images/t" + str(t) + ".tif"
        output_directory = "results/results_pairs/backward_" + str(t+1)
        result_image = register_command(moving_image_filename, fixed_image_filename, output_directory)
        np.savez("results/npz/t" + str(t+1) + "_backward.npz", result_image)

        # Forward registration
        moving_image_filename = "results/resized/test/images/t"+ str(t) + ".tif"
        fixed_image_filename = "results/resized/test/images/t" + str(t+1) + ".tif"
        output_directory = "results/results_pairs/forward_" + str(t)
        result_image = register_command(moving_image_filename, fixed_image_filename, output_directory)
        np.savez("results/npz/t" + str(t) + "_forward.npz", result_image)

###################################################################################################################

if __name__ == "__main__":


    register_pairs()


