import cv2
import numpy as np
from tqdm import tqdm

# Load left and right stereo images using OpenCV
left_image = cv2.imread('im2.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('im6.png', cv2.IMREAD_GRAYSCALE)

# Check if the images were loaded successfully
if left_image is None or right_image is None:
    print("Error: Could not load stereo images.")
else:
    # Convert the images to NumPy arrays
    left_image = np.array(left_image)
    right_image = np.array(right_image)

    # Stereo Gradient-Based Matching parameters
    window_size = 5  # Size of the matching window
    max_disparity = 64

    # Calculate gradients of the left and right images
    grad_x_left = np.gradient(left_image, axis=1)
    cv2.imshow('Disparity Map', grad_x_left)
    cv2.waitKey(0)
    
    grad_y_left = np.gradient(left_image, axis=0)
    cv2.imshow('Disparity Map', grad_y_left)
    cv2.waitKey(0)
    grad_x_right = np.gradient(right_image, axis=1)
    cv2.imshow('Disparity Map', grad_x_right)
    cv2.waitKey(0)
    grad_y_right = np.gradient(right_image, axis=0)
    cv2.imshow('Disparity Map', grad_y_right)
    cv2.waitKey(0)

    # Initialize the disparity map
    disparity = np.zeros_like(left_image, dtype=np.float32)

    half_window = window_size // 2

    for y in tqdm(range(half_window, left_image.shape[0] - half_window), desc="Processing Rows"):
        for x in tqdm(range(half_window + max_disparity, left_image.shape[1] - half_window), leave=True):
            left_patch_x = grad_x_left[y - half_window:y + half_window + 1,
                                       x - half_window:x + half_window + 1]
            left_patch_y = grad_y_left[y - half_window:y + half_window + 1,
                                       x - half_window:x + half_window + 1]

            best_match = None
            best_cost = float('inf')

            for d in range(max_disparity):
                right_x = x - d

                if right_x < half_window:
                    break

                right_patch_x = grad_x_right[y - half_window:y + half_window + 1,
                                            right_x - half_window:right_x + half_window + 1]
                right_patch_y = grad_y_right[y - half_window:y + half_window + 1,
                                            right_x - half_window:right_x + half_window + 1]

                cost = np.sum(np.abs(left_patch_x - right_patch_x) +
                              np.abs(left_patch_y - right_patch_y))

                if cost < best_cost:
                    best_cost = cost
                    best_match = d

            disparity[y, x] = best_match

    # Normalize the disparity map for visualization
    min_disp = 0
    num_disp = max_disparity
    disparity = (disparity - min_disp) / num_disp

    # Display the disparity map using OpenCV
    cv2.imshow('Disparity Map', (disparity * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
