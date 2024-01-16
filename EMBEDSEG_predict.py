import numpy as np
from tifffile import imsave, imread
from EmbedSeg.utils.create_dicts import create_test_configs_dict
from EmbedSeg.test import begin_evaluating
import os
import json

def normalize(file_location):
    all_filenames = list(os.listdir(file_location))

    for filename in all_filenames:
        img = imread(file_location + "/" + filename)
        img = 255.0 * img / np.max(img)
        imsave(file_location + "/" + filename, img)


   
data_dir = os.getcwd()
data_dir = data_dir.replace(os.sep, '/')
data_dir = data_dir + "/"
print (data_dir)
project_name = 'resized/'

normalize(data_dir + project_name + "/test/images/")

print("Evaluation images shall be read from: {}".format(os.path.join(data_dir, project_name)))

# use the following for the pretrained model weights
checkpoint_path = data_dir + '/segmentation models/full_model/checkpoint.pth'
if os.path.isfile(data_dir + 'segmentation models//data_properties.json'): 
    with open(data_dir + 'segmentation models/data_properties.json') as json_file:
        data = json.load(json_file)
        one_hot = data['one_hot']
        data_type = data['data_type']
        min_object_size = int(data['min_object_size'])
        foreground_weight = float(data['foreground_weight'])
        n_z, n_y, n_x = int(data['n_z']),int(data['n_y']), int(data['n_x'])
        pixel_size_z_microns, pixel_size_y_microns, pixel_size_x_microns = float(data['pixel_size_z_microns']), float(data['pixel_size_y_microns']), float(data['pixel_size_x_microns']) 

        avg_background_intensity = float(data['avg_background_intensity'])



tta = True
ap_val = 0.5
seed_thresh = 0.0
save_dir = './inference'
save_images = True
save_results = True
normalization_factor = 255 #65535 if data_type=='16-bit' else 255
mask_intensity = avg_background_intensity/normalization_factor

if os.path.exists(checkpoint_path):
    print("Trained model weights found at : {}".format(checkpoint_path))
else:
    print("Trained model weights were not found at the specified location!")


test_configs = create_test_configs_dict(data_dir = os.path.join(data_dir, project_name),
                                        checkpoint_path = checkpoint_path,
                                        tta = tta, 
                                        ap_val = ap_val,
                                        seed_thresh = seed_thresh, 
                                        min_object_size = min_object_size, 
                                        save_images = save_images,
                                        save_results = save_results,
                                        save_dir = save_dir,
                                        normalization_factor = normalization_factor,
                                        one_hot = one_hot,
                                        n_z = n_z,
                                        n_y = n_y,
                                        n_x = n_x,
                                        anisotropy_factor = pixel_size_z_microns/pixel_size_x_microns,
                                        name = '3d'
                                        )


begin_evaluating(test_configs, verbose = False, mask_region = None, mask_intensity = mask_intensity, avg_bg = avg_background_intensity/normalization_factor)
