import numpy as np
from tqdm import tqdm
import cv2 as cv

def stereo_matching(left_img, right_img, window_size=5):
    # Define some parameters (you should set these based on your camera calibration)
    focal_metric = 14.67e-3
    pixel_size = 12e-6
    focal_pixel = focal_metric/pixel_size  # Focal length in pixels
    baseline = 0.42   # Baseline between the two cameras in arbitrary units

    height, width = left_img.shape
    
    half_window = window_size // 2
    disparity_map = np.zeros((height, width), dtype=np.float32)

    for y in tqdm(range(half_window, height - half_window)):
        
        for x in range(half_window, width - half_window):
            left_patch = left_img[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
            best_disparity = 0
            best_corr = -1
            
            normalised_left = left_patch / np.linalg.norm(left_patch)

            for x2 in range(x, half_window, - 1):
                right_patch = right_img[y - half_window:y + half_window + 1, x2 - half_window:x2 + half_window + 1]

                
                normalised_right = right_patch / np.linalg.norm(right_patch)
                
                corr = np.sum(np.multiply(normalised_left, normalised_right))

                if corr > best_corr:
                    best_corr = corr
                    best_disparity = x - x2
        
            if best_corr < 0.1:
                best_disparity = 0
        
            # Compute depth from disparity
            if best_disparity != 0:
                depth = (focal_pixel * baseline) / best_disparity
                disparity_map[y, x] = depth

    return disparity_map

# Load your left and right images here
left_image = cv.imread("LeftNavCam.jpg", cv.IMREAD_GRAYSCALE)
left_image = cv.resize(left_image, (0, 0), fx = 0.2, fy = 0.2)
cv.imshow("Left image", left_image)

right_image = cv.imread("RightNavCam.jpg", cv.IMREAD_GRAYSCALE)
right_image = cv.resize(right_image, (0, 0), fx = 0.2, fy = 0.2)
cv.imshow("Right image", right_image)
cv.waitKey(0)
# Call the stereo_matching function
disparity_map = stereo_matching(left_image, right_image, window_size=5)

# Display or save the disparity map
# You can use libraries like OpenCV or Matplotlib for this purpose

# Example to display the disparity map using Matplotlib
import matplotlib.pyplot as plt
plt.imshow(disparity_map, cmap='plasma')
plt.colorbar()
plt.show()
