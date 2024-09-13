import numpy as np
from tifffile import imread, imsave
from scipy.spatial import cKDTree
import networkx as nx
from random import choices
import matplotlib.pyplot as plt
import os

if not os.path.exists("overseg_fine"):
    os.mkdir("overseg_fine")

start_t = 1101
end_t = 1115

for t in range(start_t, end_t):
    print (t)

    # Define the input arguments
    filename = r"C:\Users\stamatov\Desktop\Lab\RPE1 1770\overseg/t" + str(t) + ".tif"
    N_random_points = 100

    # Read an oversegmented image
    img = imread(filename)
    result = np.copy(img)
    size_threshold = 100

    for iteration in range(2):

        # Find all labels
        all_labels = list(np.unique(img))
        all_labels.remove(0)
        max_label = np.max(all_labels)


        # Iterate over them
        for label_i, obj in enumerate(all_labels):

            # find the set of points within this label
            z, y, x = np.where(img == obj);
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

        img = result
        
    imsave("overseg_fine/t" + str(t) + ".tif", img)
    
