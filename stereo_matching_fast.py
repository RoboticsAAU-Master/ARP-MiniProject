import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def preprocess_frame(frame):
    # Equalize histogram to improve contrast
    frame = cv2.equalizeHist(frame)

    # Apply a Gaussian blur to reduce noise while preserving edges
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
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
        range(num_disparities), desc="Computing disparity (gradients)", leave=False
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

    return disparity


def calc_disparity(left_image, right_image, num_disparities=64, window_size=15):
    window = np.ones([window_size, window_size]) / (window_size**2)

    # Initialize disparity maps
    disparity_maps = np.zeros(
        [left_image.shape[0], left_image.shape[1], num_disparities]
    )

    for disparity in tqdm(
        range(num_disparities), desc="Computing disparity            ", leave=False
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

    return disparity


def disparity_to_depth(
    disparity, baseline, focal_length_pixels, lower_bound=10, upper_bound=130
):
    depth = np.full_like(
        disparity, -1, dtype=np.float32
    )  # Initialize with maximum depth value (infinity)
    valid_pixels = (disparity >= lower_bound) & (disparity <= upper_bound)
    depth[valid_pixels] = (baseline * focal_length_pixels) / disparity[valid_pixels]
    return depth


if __name__ == "__main__":
    # Define stereo parameters
    num_disparities = 16 * 7  # number of disparities to check (Dont change its fine)
    window_size = 51  # block size to match (Maybe make small adjustments)
    baseline = 0.42  # baseline between the two cameras in m
    focal_length = 14.67e-3  # focal length of camera in m
    pixel_size = 12e-6 * 4  # pixel size of camera in m
    focal_length_pixels = focal_length / pixel_size  # focal length in pixels

    # left_image_path = "Images/LeftNavCam.jpg"
    # right_image_path = "Images/RightNavCam.jpg"
    left_image_path = "LFT_03_000750.tif"
    right_image_path = "RGT_03_000750.tif"
    left_image_raw = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image_raw = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    left_image_rz = cv2.resize(left_image_raw, (0, 0), fx=0.25, fy=0.25)
    right_image_rz = cv2.resize(right_image_raw, (0, 0), fx=0.25, fy=0.25)

    left_image = preprocess_frame(left_image_rz)
    right_image = preprocess_frame(right_image_rz)

    # Disparity without using gradients
    disp = calc_disparity(left_image, right_image, num_disparities, window_size)
    depth = disparity_to_depth(disp, baseline, focal_length_pixels)

    # Disparity using gradients
    disp_gd = calc_disparity_grad(left_image, right_image, num_disparities, window_size)
    depth_gd = disparity_to_depth(disp_gd, baseline, focal_length_pixels)

    # Disparity using OpenCV
    stereo = cv2.StereoBM_create(num_disparities, 11)
    disp_cv = stereo.compute(left_image, right_image)

    # Plot limits
    vmin_disp = 0
    vmax_disp = np.max([np.max(disp), np.max(disp_gd)])
    vmin_dist = -1
    vmax_dist = np.max([np.max(depth), np.max(depth_gd)])

    # Normalize cv output
    disp_cv = np.multiply(disp_cv, vmax_disp / disp_cv.max())
    depth_cv = disparity_to_depth(disp_cv, baseline, focal_length_pixels)

    # Create individual figures and axes
    fig1, ax1 = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    fig3, ax3 = plt.subplots(1, 1)
    fig4, ax4 = plt.subplots(1, 1)
    fig5, ax5 = plt.subplots(1, 1)
    fig6, ax6 = plt.subplots(1, 1)
    fig7, ax7 = plt.subplots(1, 1)

    # Plot the data on each individual subplot
    ax1.imshow(disp[:, num_disparities:], cmap="gray", vmin=vmin_disp, vmax=vmax_disp)
    ax2.imshow(depth[:, num_disparities:], cmap="gray", vmin=vmin_dist, vmax=vmax_dist)
    ax3.imshow(
        disp_gd[:, num_disparities:], cmap="gray", vmin=vmin_disp, vmax=vmax_disp
    )
    ax4.imshow(
        depth_gd[:, num_disparities:], cmap="gray", vmin=vmin_dist, vmax=vmax_dist
    )
    ax5.imshow(disp_cv[:, num_disparities:], cmap="gray")
    ax6.imshow(depth_cv[:, num_disparities:], cmap="gray")
    ax7.imshow(left_image, cmap="gray")

    # Create colorbars for each subplot
    cbar_disp1 = fig1.colorbar(ax1.images[0], shrink=0.8)
    cbar_dist2 = fig2.colorbar(ax2.images[0], shrink=0.8)
    cbar_disp_gd3 = fig3.colorbar(ax3.images[0], shrink=0.8)
    cbar_dist_gd4 = fig4.colorbar(ax4.images[0], shrink=0.8)
    cbar_disp_cv5 = fig5.colorbar(ax5.images[0], shrink=0.8)
    cbar_dist_cv6 = fig6.colorbar(ax6.images[0], shrink=0.8)

    # Set titles
    ax1.set_title("Disparity without edge enhancement")
    ax2.set_title("Distance without edge enhancement")
    ax3.set_title("Disparity with edge enhancement")
    ax4.set_title("Distance with edge enhancement")
    ax5.set_title("Disparity using OpenCV")
    ax6.set_title("Distance using OpenCV")
    ax7.set_title("Pre-processed left image")

    # Set colorbar labels
    cbar_disp1.set_label("Disparity [px]")
    cbar_dist2.set_label("Depth [m]")
    cbar_disp_gd3.set_label("Disparity [px]")
    cbar_dist_gd4.set_label("Depth [m]")
    cbar_disp_cv5.set_label("Disparity [px]")
    cbar_dist_cv6.set_label("Depth [m]")

    # Save the figures
    fig1.savefig("disparity_no_grad.png")
    fig2.savefig("distance_no_grad.png")
    fig3.savefig("disparity_grad.png")
    fig4.savefig("distance_grad.png")
    fig5.savefig("disparity_cv.png")
    fig6.savefig("distance_cv.png")
    fig7.savefig("preprocessed_image.png")

    # Show the plot
    plt.show()
