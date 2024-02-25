import requests
import cv2
import numpy as np
import json
import torch
import os
from PIL import Image
from io import BytesIO
from tqdm import tqdm

import ssl

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

#model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")


if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


def load_np_image(image):
    img =  np.array(Image.open(BytesIO(requests.get(image['image']).content)))
    return img


def predict_depth(image):
    im_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    input_batch = transform(im_bgr).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

    output = prediction.cpu().numpy()
    return output.squeeze()


with open('image_data.json', 'r') as json_file:
    data = json.load(json_file)

bar = tqdm(total=len(data))

folder_path = os.path.join(os.getcwd(), 'depth_maps')
for painting in list(data.keys()):
    image_path = os.path.join(folder_path, painting)
    if os.path.exists(f'{image_path}.npy'):
        bar.update(1)
        continue

    
    try: 
        image = load_np_image(data[painting])
        depth_map = predict_depth(image)
        depth_norm = ((depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255).astype(np.uint8)
    except KeyboardInterrupt:
        break

    try:
        np.save(image_path, depth_norm)
    except KeyboardInterrupt:
        break

    bar.update(1)
  

bar.close()


