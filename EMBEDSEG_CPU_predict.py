import numpy as np
import torch
from tifffile import imread, imwrite
import os
import erfnet
from Net_3d import *
from utils import Cluster_3d
from test_time_augmentation import apply_tta_3d
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(file_location):
    all_filenames = list(os.listdir(file_location))

    for filename in all_filenames:
        img = imread(file_location + "/" + filename)
        img = 255.0 * img / np.max(img)
        imwrite(file_location + "/" + filename, img)


if not os.path.exists("results/inference"):
    os.mkdir("results/inference")
if not os.path.exists("results/inference/predictions"):
    os.mkdir("results/inference/predictions")
   
data_dir = os.getcwd()
data_dir = data_dir.replace(os.sep, '/')
data_dir = data_dir + "/"
print (data_dir)

model_path = data_dir + '/segmentation models/full_model/best_iou_model.pth'

project_name = 'results/resized/'

normalize(data_dir + project_name + "/test/images/")

checkpoint = torch.load(model_path)
model = BranchedERFNet_3d(num_classes=[6, 1], input_channels=1).to(device)  # Adjust num_classes and input_channels as needed

new_state_dict = {}
for key, value in checkpoint['model_state_dict'].items():
    # Remove the 'module.' prefix
    new_key = key.replace('module.', '')
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict, strict = True)
#model.eval()

cluster = Cluster_3d(grid_z = 64, grid_y = 256, grid_x = 256,
                     pixel_z = 0.5823, pixel_y = 1, pixel_x = 1, device = device)

file_location = data_dir + project_name + "/test/images/"

all_filenames = list(os.listdir(file_location))

for filename in all_filenames:
    img = imread(file_location + "/" + filename)

    input_tensor = torch.from_numpy(img).float().to(device)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor

    # Direct model prediction without TTA
    with torch.no_grad():
        output = model(input_tensor)
    
    instance_map = cluster.cluster_local_maxima(
                output[0],
                n_sigma = 3,
                fg_thresh = 0.9,
                min_mask_sum = 0,
                min_unclustered_sum = 0,
                min_object_size = 1,
            )

    imwrite("results/inference/predictions/" + filename,
           instance_map.cpu().detach().numpy().astype(np.uint16))
    


