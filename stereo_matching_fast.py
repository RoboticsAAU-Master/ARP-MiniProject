import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
from tqdm import tqdm


def preprocess_frame(frame):
    frame = cv2.equalizeHist(frame)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame


def shift_image(image, shift):
    # shift image
    translation_matrix = np.float32([[1, 0, shift], [0, 1, 0]])
    shifted_image = cv2.warpAffine(
        image,
        translation_matrix,
        (image.shape[1], image.shape[0]),
    )
    return shifted_image


def calculate_disparity_gradients(
    left_image, right_image, num_disparities=64, window_size=15
):
    window = np.ones([window_size, window_size]) / window_size

    disparity_maps = np.zeros(
        [left_image.shape[0], left_image.shape[1], num_disparities]
    )

    # Compute gradients to enhance edge awareness
    grad_x_left = np.gradient(left_image, axis=1)
    grad_y_left = np.gradient(left_image, axis=0)
    grad_x_right = np.gradient(right_image, axis=1)
    grad_y_right = np.gradient(right_image, axis=0)

    for d in tqdm(range(0, num_disparities)):
        # shift right image
        grad_x_right_shifted = shift_image(grad_x_right, d)
        grad_y_right_shifted = shift_image(grad_y_right, d)

        # calculate squared differences
        SAD_x = abs(np.float32(grad_x_left) - np.float32(grad_x_right_shifted))
        SAD_y = abs(np.float32(grad_y_left) - np.float32(grad_y_right_shifted))
        SAD = SAD_x + SAD_y

        # convolve with kernel and find SAD at each point
        filtered_image = cv2.filter2D(SAD, -1, window)

        disparity_maps[:, :, d] = filtered_image

    disparity = np.argmin(disparity_maps, axis=2)
    disparity = np.uint8(disparity * 255 / num_disparities)
    disparity = cv2.equalizeHist(disparity)

    return disparity


def calculate_disparity(left_image, right_image, num_disparities=64, window_size=15):
    window = np.ones([window_size, window_size]) / window_size

    disparity_maps = np.zeros(
        [left_image.shape[0], left_image.shape[1], num_disparities]
    )

    for d in tqdm(range(0, num_disparities)):
        # shift right image
        right_image_shifted = shift_image(right_image, d)

        # calculate squared differences
        SAD = abs(np.float32(left_image) - np.float32(right_image_shifted))

        # convolve with kernel and find SAD at each point
        filtered_image = cv2.filter2D(SAD, -1, window)

        disparity_maps[:, :, d] = filtered_image

    disparity = np.argmin(disparity_maps, axis=2)
    disparity = np.uint8(disparity * 255 / num_disparities)
    disparity = cv2.equalizeHist(disparity)

    return disparity


# Define stereo parameters
num_disparities = 64  # number of disparities to check
window_size = 15  # block size to match

# left_image_path = "Images/LeftNavCam.jpg"
# right_image_path = "Images/RightNavCam.jpg"
left_image_path = "scene1.row3.col1.ppm"
right_image_path = "scene1.row3.col3.ppm"
left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
left_image = preprocess_frame(left_image)
right_image = preprocess_frame(right_image)

# Plot for comparison
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Disparity without using gradients
disparity = calculate_disparity(left_image, right_image, num_disparities, window_size)
ax[0].imshow(disparity, cmap="gray")

# Disparity using gradients
disparity = calculate_disparity_gradients(
    left_image, right_image, num_disparities, window_size
)
ax[1].imshow(disparity, cmap="gray")

# Disparity using OpenCV
stereo = cv2.StereoBM_create(num_disparities, window_size)
disparity = stereo.compute(left_image, right_image)
ax[2].imshow(disparity, cmap="gray")

# Show the plot
plt.waitforbuttonpress()
