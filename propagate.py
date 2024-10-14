
from tifffile import imread, imsave
import numpy as np

import tensorflow as tf
import voxelmorph as vxm
import sys
import os
import json

import sys
import itk
from shutil import copyfile, SameFileError

##############################################################################################################################

def instance_optimization(flow, moving, fixed, t, int_resolution = 1, epochs = 1000):
  """ Instance optimization of the deformation field. 
      Example in https://github.com/voxelmorph/voxelmorph/blob/dev/scripts/tf/train_instance.py """

  # Build an object of the instance model and set the already computed flow
  instance_model = vxm.networks.InstanceDense(inshape, int_resolution = 1)
  instance_model.set_flow(flow)

  # losses and loss weights
  losses = ['mse', vxm.losses.Grad('l2').loss]
  loss_weights = [1, 0.05]

  zeros = np.zeros((1, *inshape, len(inshape)), dtype='float32')

  instance_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4), loss = losses, loss_weights = loss_weights)
  
  if not os.path.exists("registration models/instance/instance_" + str(t) + ".h5"):

      instance_model.fit(
          [moving],
          [fixed, zeros],
          batch_size = None,
          epochs = epochs,
          steps_per_epoch = 1,
          verbose = 1)

      instance_model.save_weights("registration models/instance/instance_" + str(t) + ".h5")

  
  else:
      instance_model.load_weights("registration models/instance/instance_" + str(t) + ".h5")
  
  return instance_model

###########################################################################################################################

def build_vxm_model():
    inshape = (64, 256, 256)

    # build model using VxmDense
    # configure unet features 
    nb_features = [
        [32, 32, 32, 32],         # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]
    inshape = (64, 256, 256)
    int_steps = 7
    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps = int_steps, int_downsize = 1)

    # losses and loss weights
    losses = ['mse', vxm.losses.Grad('l2').loss]
    loss_weights = [1, 0.01]

    vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)
    vxm_model.load_weights("registration models/model_0.02_adjusted.h5")

    return vxm_model


###########################################################################################################################

def parse_manual_corrections(segmented_folder, corrected_folder, start_t, end_t):
    """ After previous passes of the tracking, the first error for each chromosome has been corrected.
        Usually, this is enough to continue propagating this chromosome correctly, as errors are spurious, not systematic. """

    corrections_per_label = {}
    
    for t in range(start_t, end_t):

        if not os.path.exists(segmented_folder + "/t" + str(t) + ".tif"):
            continue

        if not os.path.exists(corrected_folder + "/t" + str(t) + ".tif"):
            continue
        
        # Open the two images
        segmented = imread(segmented_folder + "/t" + str(t) + ".tif")
        corrected = imread(corrected_folder + "/t" + str(t) + ".tif")

        # Find the objects without the background
        objects = list(np.unique(corrected))
        objects.remove(0)

        # Collect the objects which are different between the two images - those are the ones corrected
        for obj in objects:

            if obj not in corrections_per_label:
                corrections_per_label[obj] = []
                
            points_in_segmented = list(segmented[segmented == obj])
            points_in_corrected = list(corrected[corrected == obj])

            if points_in_segmented != points_in_corrected:
                corrections_per_label[obj].append(t)

    return corrections_per_label

###########################################################################################################################

def adjust_segmentation(t, transformed_segmentation, next_segmentation, merging_threshold, embedseg = None, assigned = [], log = False, manual_correction = False, null_out = True):
    
    # Perform IoU analysis
    from collections import Counter

    assigned = []
    
    # Read in the oversegmentation and get the list of labels, removing the background
    labels = list(np.unique(next_segmentation))
    if 0 in labels:
        labels.remove(0)

    # Build the result, initially 0
    result = np.copy(transformed_segmentation)

    num_merged = 0
    
    # Find the most common color in each label
    for label in labels:


        # if this label is completely contained in an already assigned embedseg label, do nothing
        if embedseg is not None:
          points_inside = list(embedseg[next_segmentation == label])
          if np.max(points_inside) == 0:
            continue
        
          points_inside = Counter([x for x in points_inside if x != 0])
          best_color, count = points_inside.most_common()[0]
          if count / np.sum(embedseg == best_color) > 0.8:
            continue
          
        
        points_inside = list(transformed_segmentation[next_segmentation == label])
        if np.max(points_inside) == 0:
            continue
        
        points_inside = Counter([x for x in points_inside if x != 0])
        
        best_color, count = points_inside.most_common()[0]

        
        count_2 = 0
        if len(points_inside.most_common()) == 1:
            result[next_segmentation == label] = best_color
            assigned.append(label)

        else:
            # Find the second most common non-zero color
            second_color, count_2 = points_inside.most_common()[1]

            
            c = 2
            while second_color == 0 and len(points_inside.most_common()) > c:
                second_color, count_2 = points_inside.most_common()[c]
                c += 1

            # how big is the second object?
            if count / (count + count_2) > merging_threshold:
                result[next_segmentation == label] = best_color
                assigned.append(label)
                num_merged += 1

    if null_out:
        result[next_segmentation == 0] = 0
    
    return result, assigned

###########################################################################################################################

def integrate_manual_corrections(result, t, manual_changes):

    corrected_result = np.zeros_like(result)

    if not os.path.exists("results/corrected/t" + str(t) + ".tif"):
        return result
    
    corrected = imread("results/corrected/t" + str(t) + ".tif")
    all_labels = list(np.unique(result))
    all_labels.remove(0)
    
    for label in all_labels:

        if len(manual_changes[label]) == 0:
            # take the label from the corrected img
            corrected_result[result == label] = result[result == label]
            #corrected_result[corrected == label] = corrected[corrected == label]

        elif t in manual_changes[label]: #t < np.min(manual_changes[label]) or t in manual_changes[label]:

            # take the label from the corrected img
            corrected_result[corrected == label] = corrected[corrected == label]

        else:
            # take the label from the result
            corrected_result[result == label] = result[result == label]

    return corrected_result

###########################################################################################################################

def propagate_one_step(vxm_model, t, direction, manual_changes, initial_image = None):
    
    if direction == "forward":
        next_time_point = t + 1

        if initial_image is not None:
            moving_image = itk.imread(initial_image, itk.F)
        else:
            moving_image = itk.imread("results/propagated/t" + str(t) + ".tif", itk.F)
            
        parameters_location = "results/results_pairs/forward_" + str(t) + "/TransformParameters.0.txt"
    else:
        next_time_point = t - 1

        if initial_image is not None:
            moving_image = itk.imread(initial_image, itk.F)

        else:
            moving_image = itk.imread("results/propagated/t" + str(t) + ".tif", itk.F)
        parameters_location = "results/results_pairs/backward_" + str(t) + "/TransformParameters.0.txt"
   
    inshape = (64, 256, 256)

    # Load the transformix result
        
    transformix_object = itk.TransformixFilter.New(moving_image)
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(parameters_location)
    transformix_object.SetTransformParameterObject(parameter_object)

    transformix_object.UpdateLargestPossibleRegion()

    transformed_segmentation = transformix_object.GetOutput()

    # Transform via the VXM model
    moving_image = np.load("results/npz/t" + str(t) + "_forward.npz")['arr_0'] 
    fixed_image = np.load("results/npz/t" + str(t+1) + ".npz")['arr_0']
      
    moving_image = moving_image.astype('float') / np.max(moving_image.astype('float'))
    fixed_image = fixed_image.astype('float') / np.max(fixed_image.astype('float'))

    image_pair = [moving_image[np.newaxis,..., np.newaxis], fixed_image[np.newaxis,..., np.newaxis]]
    prediction, flow = vxm_model.predict(image_pair)

    #instance_model = instance_optimization(flow, moving_image[np.newaxis,..., np.newaxis], fixed_image[np.newaxis,..., np.newaxis], t, epochs = 100)
    #prediction, flow = instance_model.predict(moving_image[np.newaxis,..., np.newaxis])

    transformed_segmentation = vxm.networks.Transform(inshape, interp_method = 'nearest').predict([transformed_segmentation[np.newaxis,..., np.newaxis], flow])
   
    # Save the transformed image
    imsave("temp/t" + str(t) + ".tif", transformed_segmentation)

    transformed_segmentation = np.squeeze(transformed_segmentation)
    result = transformed_segmentation
    
    # Load the next segmented image
    next_embedseg = imread("results/inference/predictions/t" + str(next_time_point) + ".tif")
    result, assigned = adjust_segmentation(t, result, next_embedseg, 0.8, log = True, manual_correction = False, null_out = False)

    # Load the next segmented image
    next_segmentation = imread("results/overseg_fine/t" + str(next_time_point) + ".tif")

    #imsave("temp/t" + str(t) + ".tif", result, imagej = True)
    result, _ = adjust_segmentation(t, result, next_segmentation, 0.0, embedseg = next_embedseg, assigned = assigned)

    result = integrate_manual_corrections(result, next_time_point, manual_changes)
                                          
    imsave("results/propagated/t" + str(next_time_point) + ".tif", result)

###########################################################################################################################
        
if __name__ == "__main__":

    start_t = int(sys.argv[1])
    end_t = int(sys.argv[2])

    initial_location = "results/inference/predictions/"

    if len(sys.argv) > 3:
        initial_location = sys.argv[3]
        
    direction = "forward" #sys.argv[4]

    manual_changes = {} #parse_manual_corrections("results/propagated", "results/corrected", start_t, end_t)

    if not os.path.exists("results/propagated"):
        os.mkdir("results/propagated")


    try:
        copyfile(initial_location + "/t" + str(start_t) + ".tif", "results/propagated/t" + str(start_t) + ".tif")
    except SameFileError:
        pass
        

    vxm_model = build_vxm_model()

    if direction == "forward":
        propagate_one_step(vxm_model, start_t, direction, manual_changes, "results/propagated/t" + str(start_t) + ".tif")

        for t in range(start_t + 1, end_t):
            print (t)
            vxm_model = build_vxm_model()
            propagate_one_step(vxm_model, t, direction, manual_changes)

    elif direction == "backward":
        vxm_model = build_vxm_model()
        propagate_one_step(vxm_model, end_t, direction, manual_changes, "results/propagated/t" + str(end_t) + ".tif")
        
        for t in reversed(range(start_t+1, end_t)):
            print (t)
            vxm_model = build_vxm_model()
            propagate_one_step(vxm_model, t, direction, manual_changes)


    
        

    

