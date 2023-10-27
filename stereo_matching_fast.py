import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def preprocess_frame(frame):
    # Equalize histogram to improve contrast
    frame = cv2.equalizeHist(frame)

    # Apply a Gaussian blur to reduce noise while preserving edges
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame


def shift_image(image, shift):
    # Shift image a certain number of pixels
    translation_matrix = np.float32([[1, 0, shift], [0, 1, 0]])
    shifted_image = cv2.warpAffine(
        image,
        translation_matrix,
        (image.shape[1], image.shape[0]),
    )
    return shifted_image


def calc_disparity_grad(left_image, right_image, num_disparities=64, window_size=15):
    window = np.ones([window_size, window_size]) / (window_size**2)

    # Initialize disparity maps
    disparity_maps = np.zeros(
        [left_image.shape[0], left_image.shape[1], num_disparities]
    )

    # Compute gradients to enhance edge awareness
    grad_x_left = np.gradient(left_image, axis=1)
    grad_y_left = np.gradient(left_image, axis=0)
    grad_x_right = np.gradient(right_image, axis=1)
    grad_y_right = np.gradient(right_image, axis=0)

    for disparity in tqdm(
        range(num_disparities), desc="Computing disparity (gradients)"
    ):
        # Shift right image
        grad_x_right_shifted = shift_image(grad_x_right, disparity)
        grad_y_right_shifted = shift_image(grad_y_right, disparity)

        # Calculate Sum of Absolute Differences (SAD)
        SAD_x = abs(np.float32(grad_x_left) - np.float32(grad_x_right_shifted))
        SAD_y = abs(np.float32(grad_y_left) - np.float32(grad_y_right_shifted))
        SAD = SAD_x + SAD_y

        # Convolve with kernel and find SAD at each region
        filtered_image = cv2.filter2D(SAD, -1, window)
        disparity_maps[:, :, disparity] = filtered_image

    # Select disparity with minimum SAD
    disparity = np.argmin(disparity_maps, axis=2)
    #disparity = np.uint16(disparity)

    return disparity


def calc_disparity(left_image, right_image, num_disparities=64, window_size=15):
    window = np.ones([window_size, window_size]) / (window_size**2)

    # Initialize disparity maps
    disparity_maps = np.zeros(
        [left_image.shape[0], left_image.shape[1], num_disparities]
    )

    for disparity in tqdm(
        range(num_disparities), desc="Computing disparity            "
    ):
        # Shift right image
        right_image_shifted = shift_image(right_image, disparity)

        # Calculate Sum of Absolute Differences (SAD)
        SAD = abs(np.float32(left_image) - np.float32(right_image_shifted))

        # Convolve with kernel and find SAD at each region
        filtered_image = cv2.filter2D(SAD, -1, window)
        disparity_maps[:, :, disparity] = filtered_image

    # Select disparity with minimum SAD
    disparity = np.argmin(disparity_maps, axis=2)
    #disparity = np.uint16(disparity)

    return disparity


def disparity_to_depth(disparity, baseline, focal_length_pixels):
    depth = np.zeros_like(disparity, dtype=np.float32)
    depth[disparity > 10] = (baseline * focal_length_pixels) / disparity[disparity > 10]
    return depth


# TODO:
# Crop disp image to remove gradient region on left side
# Eliminate depth calculation for disp values outside the range 10 - 100 (adjust range as needed)

if __name__ == "__main__":
    # Define stereo parameters
    num_disparities = 16*8  # number of disparities to check (Dont change its fine)
    window_size =51  # block size to match (Maybe make small adjustments)
    baseline = 0.42  # baseline between the two cameras in m
    focal_length = 14.67e-3  # focal length of camera in m
    pixel_size = 12e-6  # pixel size of camera in m
    focal_length_pixels = focal_length / pixel_size  # focal length in pixels

    # left_image_path = "Images/LeftNavCam.jpg"
    # right_image_path = "Images/RightNavCam.jpg"
    left_image_path = "LFT_03_000750.tif"
    right_image_path = "RGT_03_000750.tif"
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    
    left_image = cv2.resize(left_image, (0,0), fx=0.25, fy=0.25)
    right_image = cv2.resize(right_image, (0,0), fx=0.25, fy=0.25)
    
    left_image = preprocess_frame(left_image)
    right_image = preprocess_frame(right_image)

    # Disparity without using gradients
    disp = calc_disparity(left_image, right_image, num_disparities, window_size)
    depth = disparity_to_depth(disp, baseline, focal_length_pixels)

    # Disparity using gradients
    disp_gd = calc_disparity_grad(left_image, right_image, num_disparities, window_size)
    depth_gd = disparity_to_depth(disp_gd, baseline, focal_length_pixels)

    # Disparity using OpenCV
    stereo = cv2.StereoBM_create(num_disparities, window_size)
    disp_cv = stereo.compute(left_image, right_image)

    # Plot the results for comparison
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    fig.tight_layout(pad=1.0)
    ax[0, 0].set_title("Disparity without gradients")
    ax[0, 1].set_title("Disparity with gradients")
    ax[0, 2].set_title("Disparity using OpenCV")

    # Plot limits
    vmin_disp = 0
    vmax_disp = np.max([np.max(disp), np.max(disp_gd)])
    vmin_dist = 0
    vmax_dist = np.max([np.max(depth), np.max(depth_gd)])
    
    # Normalize cv output
    disp_cv = np.multiply(disp_cv, vmax_disp / disp_cv.max())
    depth_cv = disparity_to_depth(disp_cv, baseline, focal_length_pixels)

    # Create plots
    ax[0, 0].imshow(disp, cmap="gray", vmin=vmin_disp, vmax=vmax_disp)
    ax[1, 0].imshow(depth, cmap="viridis", vmin=vmin_dist, vmax=vmax_dist)
    ax[0, 1].imshow(disp_gd, cmap="gray", vmin=vmin_disp, vmax=vmax_disp)
    ax[1, 1].imshow(depth_gd, cmap="viridis", vmin=vmin_dist, vmax=vmax_dist)
    ax[0, 2].imshow(disp_cv, cmap="gray")#, vmin=vmin_disp, vmax=vmax_disp)
    ax[1, 2].imshow(depth_cv, cmap="viridis")#, vmin=vmin_dist, vmax=vmax_dist)

    # Create colorbars for each subplot
    cbar_disp = fig.colorbar(ax[0, 0].images[0], ax=ax[0, 0], shrink=0.8)
    cbar_dist = fig.colorbar(ax[1, 0].images[0], ax=ax[1, 0], shrink=0.8)
    cbar_disp_gd = fig.colorbar(ax[0, 1].images[0], ax=ax[0, 1], shrink=0.8)
    cbar_dist_gd = fig.colorbar(ax[1, 1].images[0], ax=ax[1, 1], shrink=0.8)
    cbar_disp_cv = fig.colorbar(ax[0, 2].images[0], ax=ax[0, 2], shrink=0.8)
    cbar_dist_cv = fig.colorbar(ax[1, 2].images[0], ax=ax[1, 2], shrink=0.8)

    # Set colorbar labels
    cbar_disp.set_label("Disparity [px]")
    cbar_dist.set_label("Depth [m]")
    cbar_disp_gd.set_label("Disparity [px]")
    cbar_dist_gd.set_label("Depth [m]")
    cbar_disp_cv.set_label("Disparity [px]")
    cbar_dist_cv.set_label("Depth [m]")

    # Show the plot
    plt.show()
