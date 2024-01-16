""" The goal of this project is to segment chromosomes based on the optical flow from pairwise registration.
    The idea is that separate objects move independently. This approach is outlined in two publications:
    
    doi: 10.1109/TIP.2018.2795740 - the algorithm
    doi: 10.1007/978-3-642-15549-9_32 - constructing point trajectories

    I loosely follow their ideas here. """

import numpy as np
from tifffile import imsave, imread
import matplotlib.pyplot as plt
import sklearn.metrics
from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy.spatial import KDTree
from scipy.sparse.csgraph import shortest_path
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.spatial.distance as ssd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
import tensorflow as tf
import voxelmorph as vxm
import sys
import os
import json

##############################################################################################################################

def get_n_nearest_labels(propagated_img_3d, static_img_3d, num_neighbors, z_aspect_ratio = 1):
    """ Convert each 3D image to a list of point (N, 3).
        For each point in the static, find n nearest neighbors from the propagated."""

    # Convert each 3D image to a list of point (N, 3)
    static_img_3d_copy = np.copy(static_img_3d)

    propagated_points = np.argwhere(propagated_img_3d).astype(np.float32)
    static_points = np.argwhere(static_img_3d_copy).astype(np.float32)
    result = np.zeros_like(propagated_img_3d)

    propagated_points[:, 0] *= z_aspect_ratio
    static_points[:, 0] *= z_aspect_ratio


    # Downsample
    #propagated_points = propagated_points[::10, :]
    #static_points = static_points[::10, :]

    print (np.shape(propagated_points))
    print (np.shape(static_points))
    
    kdtree = KDTree(propagated_points)

    print ("done building the kdtree")

    
    # Find the nearest neighbors and the corresponding labels
    dist, points = kdtree.query(static_points, num_neighbors)

    print (np.shape(points))
    nearest_indices = propagated_points[points]

    for i, index in enumerate(list(nearest_indices)):

        #print (i)

        nearest_labels = []

        for p in index:
            z, y, x = p
            nearest_labels.append(propagated_img_3d[int(z/z_aspect_ratio), int(y), int(x)])
        
        counts = np.bincount(nearest_labels)
        majority_label = np.argmax(counts)

        static_p = static_points[i]
        z, y, x = static_p
        result[int(z/z_aspect_ratio), int(y), int(x)] = majority_label # TODO - assign the majority only if above threshold?

    
    return result

##############################################################################################################################

def remap_colors(segmentation, remap_file):
    """ Simply replace the colors according to the remap_schedule txt dictionary """

    remap = json.load(open(remap_file))
    for obj in remap.keys():
        segmentation[segmentation == int(obj)] = remap[obj]
    return segmentation

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

def integrate_manual_corrections(segmented_filename, corrected_filename):
    """ After previous passes of the tracking, the first error for each chromosome has been corrected.
        Usually, this is enough to continue propagating this chromosome correctly, as errors are spurious, not systematic. """

    # Open the two images
    if not os.path.exists(corrected_filename):
        return []
    
    segmented = imread(segmented_filename)
    corrected = imread(corrected_filename)

    # Find the objects without the background
    objects = list(np.unique(corrected))
    objects.remove(0)

    # Collect the objects which are different between the two images - those are the ones corrected
    changed_objects = []
    for obj in objects:
        points_in_segmented = list(segmented[segmented == obj])
        points_in_corrected = list(corrected[corrected == obj])

        if points_in_segmented != points_in_corrected:
            changed_objects.append(obj)

    return changed_objects

###########################################################################################################################

def adjust_segmentation(t, transformed_segmentation, next_segmentation, merging_threshold, log = False, manual_correction = False, null_out = True):
    
    # Perform IoU analysis
    from collections import Counter
    
    # Read in the oversegmentation and get the list of labels, removing the background
    labels = list(np.unique(next_segmentation))
    if 0 in labels:
        labels.remove(0)

    # Build the result, initially 0
    result = np.copy(transformed_segmentation)

    num_merged = 0
    print (np.shape(next_segmentation))
    
    # Find the most common color in each label
    for label in labels:
        points_inside = list(transformed_segmentation[next_segmentation == label])
        if np.max(points_inside) == 0:
            continue
        
        points_inside = Counter([x for x in points_inside if x != 0])
        
        best_color, count = points_inside.most_common()[0]

        
        count_2 = 0
        if len(points_inside.most_common()) == 1:
            result[next_segmentation == label] = best_color
            print ("perfect match")
        else:
            # Find the second most common non-zero color
            second_color, count_2 = points_inside.most_common()[1]

            
            c = 2
            while second_color == 0 and len(points_inside.most_common()) > c:
                second_color, count_2 = points_inside.most_common()[c]
                c += 1

            # how big is the second object?
            print (count_2)

            print (count / (count + count_2))
            if count / (count + count_2) > merging_threshold:
                result[next_segmentation == label] = best_color
                num_merged += 1
                print ("imperfect match")
            else:
                print ("no match")

    if null_out:
        result[next_segmentation == 0] = 0

    
    
    return result
    

###########################################################################################################################

if __name__ == "__main__":

    
    import sys
    t = int(sys.argv[1])

    datapath = "npz/"

    vxm_model = build_vxm_model()
    
    # Load a complete segmentation
    #current_segmentation = imread("tubeness/resized/t" + str(t) + ".tif")
    inshape = (64, 256, 256)

    # Run transformix offline, using results_pairs/forward_t
    #exit()

    # Load the transformix result
    transformed_segmentation = imread("temp/result.tif")

    # Transform via the VXM model
    moving_image = np.load("npz/t" + str(t) + "_forward.npz")['arr_0'] 
    fixed_image = np.load(datapath + "/t" + str(t+1) + ".npz")['arr_0']
      
    moving_image = moving_image.astype('float') / np.max(moving_image.astype('float'))
    fixed_image = fixed_image.astype('float') / np.max(fixed_image.astype('float'))

    image_pair = [moving_image[np.newaxis,..., np.newaxis], fixed_image[np.newaxis,..., np.newaxis]]
    prediction, flow = vxm_model.predict(image_pair)

    #instance_model = instance_optimization(flow, moving_image[np.newaxis,..., np.newaxis], fixed_image[np.newaxis,..., np.newaxis], t, epochs = 100)
    #prediction, flow = instance_model.predict(moving_image[np.newaxis,..., np.newaxis])

    imsave("voxelmorph_registration_result.tif", prediction)

    transformed_segmentation = vxm.networks.Transform(inshape, interp_method = 'nearest').predict([transformed_segmentation[np.newaxis,..., np.newaxis], flow])
   
    # Save the transformed image
    imsave("temp/t" + str(t) + ".tif", transformed_segmentation)

    transformed_segmentation = np.squeeze(transformed_segmentation)
    result = transformed_segmentation #get_n_nearest_labels(transformed_segmentation, next_segmentation, 40, 2)
    
    # Load the next segmented image
    next_segmentation = imread("inference/predictions/t" + str(t + 1) + ".tif")
    result = adjust_segmentation(t, result, next_segmentation, 0.9, log = True, manual_correction = False, null_out = False)

    # Load the next segmented image
    next_segmentation = imread("overseg/t" + str(t + 1) + ".tif")
    result = transformed_segmentation #get_n_nearest_labels(transformed_segmentation, next_segmentation, 40, 2)
    result = adjust_segmentation(t, result, next_segmentation, 0.0)

    
    imsave("propagated/t" + str(t+1) + ".tif", result)
        
