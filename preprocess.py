import os
import cv2
import numpy as np
from PIL import Image
import math
from scipy.signal import convolve2d
import scipy.ndimage as ndimage
from tqdm import tqdm

def normalize_data(x, low=0, high=1, data_type=None):
    x = np.asarray(x, dtype=np.float64)
    min_x, max_x = np.min(x), np.max(x)
    if max_x - min_x == 0: return x
    x = (x - float(min_x)) / float((max_x - min_x))
    x = x * (high - low) + low
    return np.asarray(x, dtype=data_type if data_type else np.float64)

def remove_hair(image, kernel_size):
    if len(image.shape) == 3: image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    return convolve2d(image, kernel, mode='same', fillvalue=0)

def compute_curvature(image, sigma):
    winsize = int(np.ceil(4 * sigma))
    window = np.arange(-winsize, winsize + 1)
    X, Y = np.meshgrid(window, window)
    G = (1.0 / (2 * math.pi * sigma ** 2)) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    G1_0 = (-X / (sigma ** 2)) * G
    G2_0 = ((X ** 2 - sigma ** 2) / (sigma ** 4)) * G
    G1_90, G2_90 = G1_0.T, G2_0.T
    hxy = ((X * Y) / (sigma ** 8)) * G
    i_g1_0, i_g2_0 = 0.1 * ndimage.convolve(image, G1_0), 10 * ndimage.convolve(image, G2_0)
    i_g1_90, i_g2_90 = 0.1 * ndimage.convolve(image, G1_90), 10 * ndimage.convolve(image, G2_90)
    fxy = ndimage.convolve(image, hxy)
    i_g1_45, i_g1_m45 = 0.5*np.sqrt(2)*(i_g1_0+i_g1_90), 0.5*np.sqrt(2)*(i_g1_0-i_g1_90)
    i_g2_45, i_g2_m45 = 0.5*i_g2_0+fxy+0.5*i_g2_90, 0.5*i_g2_0-fxy+0.5*i_g2_90
    return np.dstack([(i_g2_0/((1+i_g1_0**2)**1.5)), (i_g2_90/((1+i_g1_90**2)**1.5)), 
                      (i_g2_45/((1+i_g1_45**2)**1.5)), (i_g2_m45/((1+i_g1_m45**2)**1.5))])

def binaries(G):
    valid = G[G > 0]
    return (G > np.median(valid)).astype(np.float64) if len(valid) > 0 else np.zeros_like(G)

def connect_profile_1d(vp):
    return np.amin([np.amax([vp[3:-1], vp[4:]], axis=0), np.amax([vp[1:-3], vp[:-4]], axis=0)], axis=0)

def connect_centres(vein_score):
    connected_center = np.zeros(vein_score.shape, dtype='float64')
    vein_score_sum = np.sum(vein_score, axis=2)
    
    for index in range(vein_score_sum.shape[0]):
        connected_center[index, 2:-2, 0] = connect_profile_1d(vein_score_sum[index, :])

    for index in range(vein_score_sum.shape[1]):
        connected_center[2:-2, index, 1] = connect_profile_1d(vein_score_sum[:, index])

    i, j = np.indices(vein_score_sum.shape)
    border = np.zeros((2,), dtype='float64')
    for index in range(-vein_score_sum.shape[0] + 5, vein_score_sum.shape[1] - 4):
        connected_center[:, :, 2][i == (j - index)] = np.hstack([border, connect_profile_1d(vein_score_sum.diagonal(index)), border])

    Vud = np.flipud(vein_score_sum)
    for index in range(-vein_score_sum.shape[0] + 5, vein_score_sum.shape[1] - 4):
        mask = (i == (j - index))
        connected_center[:, :, 3][np.flipud(mask)] = np.hstack([border, connect_profile_1d(Vud.diagonal(index)), border])

    return connected_center

def profile_score_1d(p):
    t = (p > 0).astype(int)
    d = t[1:] - t[:-1]
    starts, ends = np.argwhere(d > 0).flatten() + 1, np.argwhere(d < 0).flatten() + 1
    if t[0]: starts = np.insert(starts, 0, 0)
    if t[-1]: ends = np.append(ends, len(p))
    s = np.zeros_like(p)
    for start, end in zip(starts, ends):
        chunk = p[int(start):int(end)]
        if len(chunk) > 0: s[int(start) + np.argmax(chunk)] = np.max(chunk) * (end - start)
    return s

def compute_vein_score(k):
    score = np.zeros(k.shape, dtype='float64')
    for index in range(k.shape[0]):
        score[index, :, 0] += profile_score_1d(k[index, :, 0])
    for index in range(k.shape[1]):
        score[:, index, 1] += profile_score_1d(k[:, index, 1])
    i, j = np.indices(k.shape[:2])
    for index in range(-k.shape[0] + 1, k.shape[1]):
        score[i == (j - index), 2] += profile_score_1d(k[:, :, 2].diagonal(index))
    curve_m45 = np.flipud(k[:, :, 3])
    score_m45 = np.zeros_like(curve_m45)
    for index in range(-k.shape[0] + 1, k.shape[1]):
        score_m45[i == (j - index)] += profile_score_1d(curve_m45.diagonal(index))
    score[:, :, 3] = np.flipud(score_m45)
    return score

def vein_pattern_extraction(image):
    data = np.asarray(image, dtype=np.float64)
    f = remove_hair(data, 6)
    p = normalize_data(f, 0, 255)
    kappa = compute_curvature(p, sigma=8)
    score = compute_vein_score(kappa)
    conect = connect_centres(score)
    threshold = binaries(np.amax(conect, axis=2))
    return np.multiply(image, threshold, dtype=np.float64), threshold

def main():
    source_dir = os.path.join("data", "Raw")
    target_dir = os.path.join("data", "Processed")
    
    if not os.path.exists(source_dir):
        print(f"Error: {source_dir} not found.")
        return
        
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    
    print(f"Processing {len(files)} images...")
    
    for filename in tqdm(files):
        # Use a more unique name that includes L/R as they are different patterns
        user_name = filename.replace('_1.png', '').replace('.png', '').replace(' ', '_')
        
        user_dir = os.path.join(target_dir, user_name)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            
        img_path = os.path.join(source_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_resized = cv2.resize(img, (224, 224))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        processed, mask = vein_pattern_extraction(gray)
        
        # Save the mask (vein pattern) instead of the masked image
        # This focuses the model entirely on the vein structure
        final_img = np.stack((mask * 255,) * 3, axis=-1).astype(np.uint8)
        
        save_path = os.path.join(user_dir, filename)
        cv2.imwrite(save_path, final_img)

if __name__ == "__main__":
    main()
