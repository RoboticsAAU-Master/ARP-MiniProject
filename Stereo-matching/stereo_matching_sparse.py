import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_sobel_kernels(ksize):
    if ksize % 2 == 0:
        raise ValueError("Kernel size (ksize) must be an odd number.")

    if ksize < 3:
        raise ValueError("Kernel size (ksize) must be at least 3.")

    half_ksize = ksize // 2

    # Initialize the kernels with zeros
    kernel_x = np.zeros((ksize, ksize))
    kernel_y = np.zeros((ksize, ksize))

    for i in range(ksize):
        for j in range(ksize):
            if i < half_ksize:
                kernel_x[j, i] = -(half_ksize - i)
                kernel_y[i, j] = -(half_ksize - i)
            elif i == half_ksize:
                kernel_x[j, i] = 0
                kernel_y[i, j] = 0
            else:
                kernel_x[j, i] = i - half_ksize
                kernel_y[i, j] = i - half_ksize

    return kernel_x, kernel_y


def sobel_edge_detection(image, ksize=7):
    # Define the Sobel kernels for horizontal and vertical edge detection
    kernel_x, kernel_y = generate_sobel_kernels(ksize)
    half_ksize = ksize // 2

    # Ensure the image is in grayscale (if not, convert it)
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.float64)

    # Initialize output arrays for horizontal and vertical gradients
    grad_x = np.zeros_like(image, dtype=np.float64)
    grad_y = np.zeros_like(image, dtype=np.float64)

    # Iterate over the image and convolve with Sobel kernels
    height, width = image.shape
    for y in tqdm(
        range(half_ksize, height - half_ksize), desc="Edges    ", leave=False
    ):
        for x in range(half_ksize, width - half_ksize):
            patch = image[
                y - half_ksize : y + half_ksize + 1, x - half_ksize : x + half_ksize + 1
            ]
            grad_x[y, x] = np.sum(patch * kernel_x)
            grad_y[y, x] = np.sum(patch * kernel_y)

    return grad_x, grad_y


def harris_corner_detection(image, k=0.04, threshold=0.01):
    # Convert the image to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate image gradients using Sobel filters
    dx, dy = sobel_edge_detection(image)

    # Compute components of the Harris matrix
    Ixx = dx * dx
    Iyy = dy * dy
    Ixy = dx * dy

    # Calculate the sum of squared differences
    window_size = 5
    offset = window_size // 2
    height, width = image.shape
    keypoints = []
    descriptors = []

    for y in tqdm(range(offset, height - offset), desc="Keypoints", leave=False):
        for x in range(offset, width - offset):
            # Compute the sum of squared differences in the window
            Sxx = np.sum(Ixx[y - offset : y + offset + 1, x - offset : x + offset + 1])
            Syy = np.sum(Iyy[y - offset : y + offset + 1, x - offset : x + offset + 1])
            Sxy = np.sum(Ixy[y - offset : y + offset + 1, x - offset : x + offset + 1])

            # Calculate the determinant and trace of the Harris matrix
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy

            # Calculate the corner response function
            r = det - k * (trace**2)

            # Apply the threshold to detect corners
            if r > threshold:
                keypoints.append([x, y])
                descriptors.append([Sxx, Syy, Sxy])

    return np.array(keypoints), np.array(descriptors)


def match_keypoints(keypoints1, descriptors1, keypoints2, descriptors2, threshold=0.7):
    matches = []

    for i in tqdm(range(len(keypoints1)), desc="Matching  ", leave=False):
        descriptor1 = descriptors1[i]

        best_match_index = -1
        best_match_score = float("-inf")

        for j in range(len(keypoints2)):
            descriptor2 = descriptors2[j]

            # Calculate the sum of squared differences (SSD) between descriptors
            ssd = np.sum((descriptor1 - descriptor2) ** 2)

            if ssd < best_match_score:
                best_match_score = ssd
                best_match_index = j

        # Check if the best match is below the threshold
        if best_match_score < threshold:
            matches.append((keypoints1[i], keypoints2[best_match_index]))

    return matches


def match_keypoints_fast(
    keypoints1, descriptors1, keypoints2, descriptors2, threshold=5e2
):
    # Calculate the pairwise Euclidean distances between descriptors
    distances = np.linalg.norm(descriptors1[:, None] - descriptors2, axis=2)

    # Find the best match index for each descriptor in descriptors1
    best_match_indices = np.argmin(distances, axis=1)
    best_match_distances = distances[np.arange(len(keypoints1)), best_match_indices]

    # Create a mask for valid matches based on the threshold
    valid_matches = best_match_distances < threshold

    # Extract the matched keypoints
    matched_keypoints = [
        (keypoints1[i], keypoints2[best_match_indices[i]])
        for i in range(len(keypoints1))
        if valid_matches[i]
    ]

    return np.array(matched_keypoints)


def match_corners(left_corners, right_corners):
    left_corners = np.array(left_corners)
    right_corners = np.array(right_corners)

    left_corners_expanded = left_corners[:, np.newaxis, :]
    distances = np.sqrt(np.sum((left_corners_expanded - right_corners) ** 2, axis=2))

    # Find the closest corner in the right image for each corner in the left image
    min_indices = np.argmin(distances, axis=1)

    matched_points = np.array(
        [left_corners[i] for i in range(len(left_corners)) if i == min_indices[i]]
    )

    return matched_points, min_indices


# test the function
if __name__ == "__main__":
    image_left_raw = cv2.imread("Input/scene1.row3.col1.ppm")
    image_right_raw = cv2.imread("Input/scene1.row3.col3.ppm")

    image_left = cv2.GaussianBlur(image_left_raw, (3, 3), 0)
    image_right = cv2.GaussianBlur(image_right_raw, (3, 3), 0)

    # Detect corners
    keypoints_left, descriptors_left = harris_corner_detection(image_left, 0.2, 0.01)
    keypoints_right, descriptors_right = harris_corner_detection(image_right, 0.2, 0.01)
    print("Number of corners detected in the left image: ", len(keypoints_left))
    print("Number of corners detected in the right image: ", len(keypoints_right))

    # matches, indecies = match_corners(keypoints_left, keypoints_right)
    matches = match_keypoints_fast(
        keypoints_left, descriptors_left, keypoints_right, descriptors_right
    )
    print("Number of corners matched: ", len(matches))

    # Create a figure with two subplots for the left and right images
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Horizontally stack the images
    stacked_image = np.hstack((image_left_raw, image_right_raw))

    # Display the stacked image
    ax.imshow(stacked_image)

    # Calculate the offset for drawing points in the right image
    offset_x = image_left_raw.shape[1]

    # Plot the left image with matched points (red)
    ax.plot(matches[:, 0, 0], matches[:, 0, 1], "r.")

    # Plot the right image with matched points (blue)
    ax.plot(matches[:, 1, 0] + offset_x, matches[:, 1, 1], "b.")

    # Set titles for subplots
    ax.set_title("Key points matched in both images")

    # Draw lines connecting matched points
    for i in range(len(matches)):
        x1, y1 = matches[i, 0]
        x2, y2 = matches[i, 1]
        x2 += offset_x
        ax.plot([x1, x2], [y1, y2], "g-")

    # Display the figure
    plt.show()

    # Draw corners on the original image and display it
    for x, y in keypoints_left:
        cv2.circle(image_left, (x, y), 0, (0, 255, 0), -1)

    plt.imshow(image_left)
    plt.show()
