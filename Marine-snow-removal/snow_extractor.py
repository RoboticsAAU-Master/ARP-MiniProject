import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from time import time
from segmentation import (
    color_segmentation,
    density_segmentation,
    superpixel_segmentation,
)

### "Marine snow detection for real time feature detection" ###
# https://doi.org/10.1109/AUV53081.2022.9965895

### This is a causal version of non-causal snow_removal_1.py ###


# Function to get index for circular array
def wrap_index(index: int) -> int:
    return index % 3


### PARAMETERS ###
# Coefficients for Y-channel (BGR)
COEFFS = np.array([0.114, 0.587, 0.299])
# Kernel for tophat morphology
kernel1 = np.ones((3, 3), np.uint8)
# Kernel for mean filtering
w1 = 50
# Threshold for mean-centered image
th_mean = 5
# Kernel for density map
w2 = 50  #50
kernel2 = np.ones((w2, w2)) / (w2**2)
# Kernel for open/dilation operation of feature mask
kernel3 = cv.getStructuringElement(cv.MORPH_RECT, (w2, w2))
# Threshold for dense regions
th_dense = 1 #5
##################


# cap = cv.VideoCapture("marine_snow.MP4")
cap = cv.VideoCapture('water_column_3.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

NUM_FRAMES = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

'''cap2 = cv.VideoCapture('water_NoSnow.mp4')
if not cap2.isOpened():
    print("Cannot open camera")
    exit()'''
    
image_directory = 'Marine-snow-removal/backround/horse_canyon/imgs'

#NUM_FRAMES = int(cap2.get(cv.CAP_PROP_FRAME_COUNT))

# Create window with freedom of dimensions
cv.namedWindow("Detection", cv.WINDOW_NORMAL)
cv.namedWindow("Snow Removed", cv.WINDOW_NORMAL)

start_time = time()
frame_counter = 0
images = []
idx = 1  # Index for accesing images

for i, image_file in enumerate(os.listdir(image_directory)):
    # Capture frame-by-frame
    image_path = os.path.join(image_directory, image_file)
    frame2 = cv.imread(image_path)
    
    ret, frame = cap.read()
    
    frame = cv.resize(frame, (968,608))
    
    image_path = os.path.join(image_directory, image_file)
    
    # Read and display the image
    #ret2, frame2 = cap2.read()
    

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    '''if not ret2:
        print("Can't receive frame (stream end?). Exiting ...")
        break'''

    frame_counter += 1
    # Converting input image to gray with weights
    Y = cv.transform(frame, COEFFS.reshape((1, 3)))
    
    gray_frame2 = cv.transform(frame2, COEFFS.reshape((1, 3)))

    # Mean filtering the gray-scale image
    Y_blurred = cv.blur(Y, (w1, w1), cv.BORDER_REPLICATE)
    # Compute mean-centered image
    mean_centering = cv.subtract(Y, Y_blurred)
    # Threshold the mean-centered image to obtain a binary mask of snow candidates
    mean_centering = cv.threshold(mean_centering, th_mean, 255, cv.THRESH_BINARY)[1]
    # Compute binary density map
    density_map = density_segmentation(mean_centering, kernel2, th_dense, kernel3)
    # Remove dense areas (features)
    snow_mask = cv.subtract(mean_centering, density_map)
    
    

    # Blob analysis
    num_labels, blobs, stats, centroids = cv.connectedComponentsWithStats(snow_mask)
    blob_mask = np.zeros_like(snow_mask)
    # Compute the area for all labeled components in one go
    blob_areas = stats[1:, cv.CC_STAT_AREA]
    # Create an array of labels to keep based on the area threshold
    remove_labels = (
        np.where(blob_areas < 15)[0] + 1
    )  # Adding 1 to account for 0-based indexing
    # Create a mask of labels to keep
    remove_mask = np.isin(blobs, remove_labels)
    # Apply the keep_mask to the original image
    blob_mask[remove_mask] = 255
    snow_mask2 = cv.subtract(snow_mask, blob_mask)
    
    P_mask_rgb = cv.merge([snow_mask2, snow_mask2, snow_mask2])
    frame2_w_snow = cv.addWeighted(frame2, 1, P_mask_rgb, 0.75, 0)
    
    frame2_w_snow = cv.GaussianBlur(frame2_w_snow, (15, 15), 0)
    
    frame2[P_mask_rgb > 0] = frame2_w_snow[P_mask_rgb > 0]
    
    
    # Copy original image
    snow_removed = Y.copy()
    # Applying blurring only in mask region
    snow_removed[snow_mask2 > 0] = Y_blurred[snow_mask2 > 0]

    # Overlay snow mask with original image
    P_mask_rgb = cv.merge([snow_mask2, snow_mask2, snow_mask2])
    P_overlay = cv.addWeighted(frame, 1, P_mask_rgb, 1, 0)

    P_overlay = cv.putText(
        P_overlay,
        f"FPS: {(1 / (time() - start_time)):.2f}",
        (0, 100),
        cv.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 0),
        2,
        cv.LINE_AA,
    )

    concatenate_detection = cv.hconcat([P_overlay, P_mask_rgb])
    concatenate_removal = cv.hconcat([Y, snow_removed])

    cv.imshow("Snow Added",frame2)
    
    mask_file_path = os.path.join('Marine-snow-removal/data/masks', f'mask_{frame_counter:04d}.png')
    frame_file_path = os.path.join('Marine-snow-removal/data/images', f'image_{frame_counter:04d}.png')
    cv.imwrite(frame_file_path, frame2)
    cv.imwrite(mask_file_path, P_mask_rgb)
    
    #cv.imshow("Detection", concatenate_detection)
    #cv.imshow("Snow Removed", concatenate_removal)

    if cv.waitKey(1) == ord("q"):
        break

    # if idx == 508:
    #     cv.imwrite("marine_snow_rgb3.png", frame)
    #     cv.imwrite("marine_snow_monochrome3.png", Y)
    #     cv.imwrite("marine_snow_rectified3.png", rectified)
    #     cv.imwrite("marine_snow_density3.png", snow_mask)
    #     cv.imwrite("marine_snow_blob3.png", snow_mask2)
    #     cv.imwrite("marine_snow_overlay3.png", P_overlay)

    start_time = time()
    # disp_frame = frame
    idx += 1

    # If the last frame is reached, reset the capture and the frame_counter
    if frame_counter == NUM_FRAMES - 1:
        frame_counter = 0  # Or whatever as long as it is the same as next line
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# When everything done, release the capture
cap.release()
#cap2.release()
cv.destroyAllWindows()
