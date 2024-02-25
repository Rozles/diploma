import numpy as np
import json
import os
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

DISTANCE_MATRIX = 'ssim_distance_matrix2.npy'

def calculate_distances(data, distance_matrix):
    n = len(data)
    keys = list(data.keys())

    bar = tqdm(total=n)  # Progress bar
    try:
    
        for i in range(n):
            for j in range(i+1, n):  # Use symmetry, matrix is mirrored along the diagonal
                if distance_matrix[i, j] == 0:
                    depth_map1 = np.load(f'depth_maps_100/{keys[i]}.npy')
                    depth_map2 = np.load(f'depth_maps_100/{keys[j]}.npy')
                    ## SSIM
                    dist = ssim(depth_map1, depth_map2, data_range=1)
                    ## Euclidean
                    # dist = np.linalg.norm(depth_map1 - depth_map2)
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist  # Symmetry

            bar.update(1)
            if (i % 10) == 0:
                np.save(DISTANCE_MATRIX, distance_matrix)

        # Final save
        np.save(DISTANCE_MATRIX, distance_matrix)
        print("Done, saving...")
        bar.close()

    except KeyboardInterrupt:
        # Save on keyboard interrupt
        np.save(DISTANCE_MATRIX, distance_matrix)
        print("Keyboard interrupt, saving...")
        bar.close()

def main():
    # Load the data
    with open('image_data.json') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} images")

    if os.path.exists(DISTANCE_MATRIX):
        distance_matrix = np.load('ssim_distance_matrix.npy')
        print("Loaded distance matrix")
        print("Missing values:", np.count_nonzero(distance_matrix == 0))
    else:
        distance_matrix = np.zeros((len(data), len(data)))
        print("Created new distance matrix")

    # Calculate the distances
    calculate_distances(data, distance_matrix)

if __name__ == "__main__":
    main()
