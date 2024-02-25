import requests
import cv2
import numpy as np
import json
import torch
import os
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from retinaface import RetinaFace

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


with open('all_data_new.json', 'r') as json_file:
    data = json.load(json_file)

folder_path = os.path.join(os.getcwd(), 'depth_maps')

bar = tqdm(total=len(data))
for id_ in data:
    painting = data[id_]
    image_path = os.path.join(folder_path, id_)
    if 'retinaface' in painting and os.path.exists(f'{image_path}.npy'):
        bar.update(1)
        continue

    image = load_np_image(painting)
    depth_map = predict_depth(image)

    if not os.path.exists(f'{image_path}.npy'):
        depth_norm = depth_norm = ((depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255).astype(np.uint8)
        np.save(image_path, depth_norm)

    if 'retinaface' in painting:
        bar.update(1)
        continue


    try: 
        x, y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))
        x_factor = depth_map.shape[1] / image.shape[1]
        y_factor = depth_map.shape[0] / image.shape[0]

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if image.shape[2] == 4:
            image = image[:, :, :3]
        
        
        faces_rf = RetinaFace.detect_faces(image)
        for face in faces_rf:
            if len(face) == 0:
                faces_rf = {}
                break
            x1, y1, x2, y2 = faces_rf[face]['facial_area'][0], faces_rf[face]['facial_area'][1], faces_rf[face]['facial_area'][2], faces_rf[face]['facial_area'][3]
            diameter = (x2 - x1) * 3 / 4
            x_on_depth_map = int((x1 + x2) / 2 * x_factor)
            y_on_depth_map = int((y1 + y2) / 2 * y_factor)
            x_i = x[np.where(np.sqrt((x - x_on_depth_map)**2 + (y - y_on_depth_map)**2) < diameter)] 
            y_i = y[np.where(np.sqrt((x - x_on_depth_map)**2 + (y - y_on_depth_map)**2) < diameter)]
            dist = np.mean(depth_map[y_i, x_i])
            faces_rf[face]['depth'] = dist
        
        painting['retinaface'] = str(faces_rf)
    except KeyboardInterrupt:
        break
        

    bar.update(1)
    # if bar.n % 1000 == 0:
    #     with open(f'checkpoint_{bar.n // 1000}.json', 'w') as outfile:
    #         json.dump(data, outfile)

bar.close()
with open('all_data_new_2.json', 'w') as outfile:
    json.dump(data, outfile)

