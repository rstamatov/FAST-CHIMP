import numpy as np
from tifffile import imread, imwrite
from collections import Counter
import os

def refine(segmentation, oversegmentation):
    oversegmentation_labels = list(np.unique(oversegmentation))
    oversegmentation_labels.remove(0)
    
    segmentation_maxlabel = segmentation.max() + 1


    result = np.zeros_like(oversegmentation)

    for label in oversegmentation_labels:
        mask = (oversegmentation == label)

        points_inside = list(segmentation[mask])
        points_inside = Counter([x for x in points_inside if x != 0])

        most_common_label = 0

        if len(points_inside.most_common()) > 0:
            most_common_label = points_inside.most_common()[0][0]

        if most_common_label != 0:  # This checks ensures we're not assigning 0 due to the earlier adjustment
            result[mask] = most_common_label
        else:
            # Assign a unique new label to avoid confusion with existing segmentation labels
            result[mask] = segmentation_maxlabel
            segmentation_maxlabel += 1
    
    return result


filenames = list(os.listdir("results/inference/predictions"))

for f in filenames:
    segmentation = imread("results/inference/predictions/" + f)
    oversegmentation = imread("results/overseg/" + f)
    result = refine(segmentation, oversegmentation)
    result = np.array(result, dtype = np.float32)
    imwrite("results/inference/predictions/" + f, result, imagej = True)
