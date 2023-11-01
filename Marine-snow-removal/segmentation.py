import numpy as np
import cv2 as cv

# Much faster than the standard class
from fast_slic.avx2 import SlicAvx2


# Segmentation based on density
def density_segmentation(
    img: np.ndarray, density_kernel: tuple, th_dense: int, morph_kernel: tuple
) -> np.ndarray:
    # Compute density map
    P_density = cv.filter2D(img, -1, density_kernel, borderType=cv.BORDER_REPLICATE)
    # Threshold to only keep dense regions corresponding to features, i.e. locate areas of many features
    _, P_density = cv.threshold(P_density, th_dense, 255, cv.THRESH_BINARY)
    # Perform opening operation to remove noise
    P_density = cv.morphologyEx(P_density, cv.MORPH_OPEN, morph_kernel)
    # Perform closing to remove holes in blobs
    P_density = cv.morphologyEx(P_density, cv.MORPH_DILATE, morph_kernel)

    return P_density


# Segmentation based on k-means clustering
def color_segmentation(
    img: np.ndarray, num_clusters: int, num_iterations: int = 10, accuracy: float = 1.0
) -> np.ndarray:
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        num_iterations,
        accuracy,
    )
    # Perform k-means clustering
    ret, label, center = cv.kmeans(
        Z, num_clusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS
    )
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2


# Segmentation based on SLIC
def superpixel_segmentation(
    img: np.ndarray, num_components: int, compactness: int
) -> np.ndarray:
    # Compute superpixels
    slic = SlicAvx2(num_components=num_components, compactness=compactness)
    assignment = slic.iterate(cv.cvtColor(img, cv.COLOR_RGB2LAB))  # Cluster Map
    # cv.imshow("Assignment", cv.multiply(assignment, 2.5).astype(np.uint8))
    # print(assignment)
    # print(slic.slic_model.clusters)  # The cluster information of superpixels.

    # Convert input image to grayscale
    frame_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Create empty snow mask
    snow_mask = np.zeros((img.shape[0], img.shape[1]))

    # Iterate over all superpixels
    for i in range(num_components):
        # Create a mask for the current superpixel
        mask = (assignment == i).astype(np.uint8)

        # Apply the mask to the gray-version of original image
        masked_image = cv.bitwise_and(frame_gray, frame_gray, mask=mask)

        # Calculate the histogram for the masked region
        hist = cv.calcHist([masked_image], [0], mask, [256], [0, 255])

        # Get the number of members in the current superpixel
        num_members = slic.slic_model.clusters[i]["num_members"]

        # Experimental thresholds
        if np.max(hist) > num_members // 4 and np.argmax(hist) < 150:
            snow_mask = cv.bitwise_or(snow_mask, mask)

    return snow_mask * 255
