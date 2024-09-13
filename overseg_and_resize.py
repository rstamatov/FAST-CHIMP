import numpy as np

from tifffile import imread, imsave

import os
from scipy.ndimage import zoom
from random import choices

# for watershed
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import extrema

from scipy.spatial import cKDTree
import networkx as nx


###########################################################################
#    Parse the arguments
###########################################################################

import sys

intensity_threshold = 50
watershed_tolerance = 0.1
size_threshold = 100
N_random_points = 100

if len (sys.argv) == 3:
    intensity_threshold = float(sys.argv[1])
    watershed_tolerance = float(sys.argv[2])

def resize (image, target_shape = (64, 256, 256)):
    Nz, Ny, Nx = np.shape(image)
    Nz_out, Ny_out, Nx_out = target_shape    
    return zoom(image, (Nz_out/Nz, Ny_out/Ny, Nx_out/Nx), prefilter = False, order = 0)

############################################################################################

def change_bit_depth(image):
    return np.array(image, dtype = np.float32)

#############################################################################################

def watershed_oversegmentation(t, image, tube, intensity_threshold, watershed_tolerance):
        
    h_maxima, _ = ndi.label(extrema.h_maxima(image, h = watershed_tolerance))

    mask = np.zeros_like(image)
    mask[tube > intensity_threshold] = 1
    labels = watershed(255-image, h_maxima, mask = mask.astype(bool))

    return resize(labels)

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

input_file_location = "results/restored/"

from os import listdir
from os.path import isfile, join
all_files = [f for f in listdir(input_file_location) if isfile(join(input_file_location, f))]


if not os.path.exists("results/resized"):
    os.mkdir("results/resized")

if not os.path.exists("results/resized/test"):
    os.mkdir("results/resized/test")

if not os.path.exists("results/resized/test/images"):
    os.mkdir("results/resized/test/images")

if not os.path.exists("results/overseg"):
    os.mkdir("results/overseg")

if not os.path.exists("results/overseg_fine"):
    os.mkdir("results/overseg_fine")

for f in all_files:
    print(f)
    t = int(f[-8:-4])
    
    img = imread(input_file_location + "/" + f)
    tube = imread("results/tubeness/t" + str(t) + ".tif")

    img = resize(img)
    imsave("results/resized/test/images/t" + str(t) + ".tif", img)
    
    tube = resize(tube)
    img[tube < intensity_threshold] = 0
  
    watershed_result = watershed_oversegmentation(t, img, tube, intensity_threshold, watershed_tolerance)
    imsave("results/overseg/t" + str(t) + ".tif", watershed_result)

    # Split the oversegmented chunks further
    result = np.copy(watershed_result)

    for iteration in range(2):

        # Find all labels
        all_labels = list(np.unique(watershed_result))
        all_labels.remove(0)
        max_label = np.max(all_labels)


        # Iterate over them
        for label_i, obj in enumerate(all_labels):

            # find the set of points within this label
            z, y, x = np.where(watershed_result == obj);
            all_points = np.vstack([z, y, x]).T
            size = len(all_points)

            if size < size_threshold:
                continue

            # Take a sample of points
            index = choices(list(range(len(x))), k = N_random_points)
            sample_points = np.vstack([z[index], y[index], x[index]]).T
            points_np = np.array(sample_points)

            # construct a graph
            graph = nx.Graph()

            for i, point in enumerate(points_np):
                graph.add_node(i)

            tree = cKDTree(points_np)
            for i, point in enumerate(points_np):
                for j in tree.query_ball_point(point, 4):
                    if i != j:  # avoid self-loops
                        distance = np.linalg.norm(points_np[i] - points_np[j])
                        graph.add_edge(i, j, weight = distance)
            

            # find the endpoints - choose a random point, find the most distant
            # and then the most distant from that one, in turn

            chosen = choices(list(graph.nodes), k = 1)[0]
            path_lengths = nx.single_source_dijkstra_path_length(graph, chosen, weight='weight')
            start, max_distance = max(path_lengths.items(), key = lambda x: x[1])
            path_lengths = nx.single_source_dijkstra_path_length(graph, start, weight='weight')
            end, max_distance = max(path_lengths.items(), key = lambda x: x[1])

            start = np.array(sample_points[start])
            end = np.array(sample_points[end])

            # Color each pixel according to the start or end node it's closer to
            max_label += 1
            new_color = max_label
            for p in all_points:
                z, y, x = p
                if np.linalg.norm(p - start) < np.linalg.norm(p - end):
                    result[z, y, x] = obj
                else:
                    result[z, y, x] = new_color

        watershed_result = result
        
    imsave("results/overseg_fine/t" + str(t) + ".tif", watershed_result)
    


