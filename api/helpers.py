import numpy as np
from sklearn.cluster import KMeans
import cv2
from PIL import Image

def compute_snr(original_data, processed_data):
    signal_power = np.mean(original_data ** 2)
    noise_power = np.mean((original_data - processed_data) ** 2)
    return 10 * np.log10(signal_power / noise_power) if noise_power != 0 else float('inf')

def apply_kmeans_compression(image_data, compression_level):
    k_values = {'low': 8, 'medium': 16, 'high': 32}
    k = k_values[compression_level]
    original_shape = image_data.shape
    img_data_flattened = image_data.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_data_flattened)
    compressed_img = kmeans.cluster_centers_[kmeans.labels_]
    compressed_img = compressed_img.reshape(original_shape).astype(np.uint8)
    return compressed_img, compute_snr(image_data, compressed_img)

def apply_noise_reduction(image_data, reduction_type, level):
    ksize_values = {'low': 3, 'medium': 5, 'high': 7}
    ksize = ksize_values[level]
    if reduction_type == 'median':
        reduced_img = cv2.medianBlur(image_data, ksize)
    else:  # gaussian
        reduced_img = cv2.GaussianBlur(image_data, (ksize, ksize), 0)
    return reduced_img, compute_snr(image_data, reduced_img)

def apply_image_sharpening(image_data, level):
    strength = {'low': -1, 'medium': -3, 'high': -5}
    kernel = np.array([[0, strength[level], 0],
                       [strength[level], 9 - 4*strength[level], strength[level]],
                       [0, strength[level], 0]])
    sharpened_img = cv2.filter2D(image_data, -1, kernel)
    return sharpened_img, compute_snr(image_data, sharpened_img)

def apply_segmentation(image_data, level):
    k_values = {'low': 2, 'medium': 4, 'high': 8}
    k = k_values[level]
    original_shape = image_data.shape
    img_data_flattened = image_data.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_data_flattened)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_img = segmented_img.reshape(original_shape).astype(np.uint8)
    return segmented_img, compute_snr(image_data, segmented_img)
