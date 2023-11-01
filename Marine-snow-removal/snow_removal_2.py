import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from time import time

### "Real-time marine snow noise removal from underwater video sequences" ###
# https://doi.org/10.1117/1.JEI.27.4.043002


# Function to get index for circular array
def wrap_index(index: int) -> int:
    return index % 3


### PARAMETERS ###
# Grayscale conversion
GRAY_MODE = True
COEFFS = np.array([0.114, 0.587, 0.299])  # YCbCr = [0.114, 0.587, 0.299]
# Temporal number of frames
t = 3
# Patch width
q = 32
# Threshold (equation 3)
c = 10
# Spatial patch size (between 1 and 7) (Should be less than q)
s = 3
r = s // 2
# Threshold (equation 7)
d = 5
# Treshold for density map (smaller means removing more)
c_density = 5
# Kernel for density map
w = 50
kernel1 = np.ones((w, w)) / (w**2)
# Kernel for open/dilation operation of feature mask
kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (w, w))
##################


cap = cv.VideoCapture("Input/Grass.MP4")
if not cap.isOpened():
    print("Cannot open camera")
    exit()

NUM_FRAMES = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# Create window with freedom of dimensions
cv.namedWindow("Original", cv.WINDOW_NORMAL)
# Create window with freedom of dimensions
cv.namedWindow("Output", cv.WINDOW_NORMAL)

start_time = time()
frame_counter = 0
images = []
idx = 1  # Index for accesing images
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_counter += 1

    if GRAY_MODE:
        frame = cv.transform(frame, COEFFS.reshape((1, 3)))
    # frame = cv.resize(frame, (0, 0), fx=0.2, fy=0.2)

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if len(images) < 3:
        images.append(frame)
        if len(images) < 3:
            disp_frame = np.zeros_like(frame)
            continue
    else:
        # Update the future image
        images[wrap_index(idx + 1)] = frame

    # Step 1-2
    prev_img = images[wrap_index(idx - 1)]
    curr_img = images[wrap_index(idx)]
    next_img = images[wrap_index(idx + 1)]

    # Step 3-4
    if GRAY_MODE:
        mask1 = np.ones_like(frame, dtype=np.uint8) * 255
    else:
        images_D = []
        for img in images:
            img_D = cv.add(
                cv.absdiff(img[:, :, 2], img[:, :, 1]),
                cv.add(
                    cv.absdiff(img[:, :, 1], img[:, :, 0]),
                    cv.absdiff(img[:, :, 0], img[:, :, 2]),
                ),
            )
            images_D.append(img_D)

        # Step 5-6
        img_criteria1 = images_D[wrap_index(idx)] - np.minimum(
            cv.blur(
                images_D[wrap_index(idx - 1)], (q, q), borderType=cv.BORDER_REPLICATE
            ),
            cv.blur(
                images_D[wrap_index(idx + 1)], (q, q), borderType=cv.BORDER_REPLICATE
            ),
        )

        # img_criteria1 = images_D[wrap_index(idx)]
        _, mask1 = cv.threshold(img_criteria1, c, 255, cv.THRESH_BINARY_INV)
        # cv.imshow("Mask1", mask1)

    # Step 7-11
    img_criteria2_prev = cv.subtract(
        curr_img, cv.dilate(prev_img, (s, s), borderType=cv.BORDER_REPLICATE)
    )
    img_criteria2_next = cv.subtract(
        curr_img, cv.dilate(next_img, (s, s), borderType=cv.BORDER_REPLICATE)
    )
    # img_criteria2_prev = cv.multiply(img_criteria2_prev, 10)
    # _, mask2_prev = cv.threshold(img_criteria2_prev, d, 255, cv.THRESH_BINARY)
    # _, mask2_next = cv.threshold(img_criteria2_next, d, 255, cv.THRESH_BINARY)
    if GRAY_MODE:
        mask2_prev = (img_criteria2_prev > d).astype(np.uint8)
        mask2_next = (img_criteria2_next > d).astype(np.uint8)
    else:
        mask2_prev = np.any(img_criteria2_prev > d, axis=-1).astype(np.uint8)
        mask2_next = np.any(img_criteria2_next > d, axis=-1).astype(np.uint8)
    mask2 = cv.multiply(cv.bitwise_and(mask2_prev, mask2_next), 255)

    combined_mask = cv.bitwise_and(mask1, mask2)
    # cv.imshow("Combined mask", combined_mask)

    # Compute density map
    P_density = cv.filter2D(combined_mask, -1, kernel1, borderType=cv.BORDER_REPLICATE)
    # Threshold to only keep dense regions corresponding to features, i.e. locate areas of many features
    _, P_density = cv.threshold(P_density, c_density, 255, cv.THRESH_BINARY)
    # Perform opening operation to remove noise
    P_density = cv.morphologyEx(P_density, cv.MORPH_OPEN, kernel2)
    # Perform closing to remove holes in blobs
    P_density = cv.morphologyEx(P_density, cv.MORPH_DILATE, kernel2)
    # Isolate snow by subtracting feature mask
    combined_mask = cv.subtract(combined_mask, P_density)
    # Remove noise with a median filter
    # combined_mask = cv.medianBlur(combined_mask, 5)

    # Step 12-14
    num_labels, blobs, stats, centroids = cv.connectedComponentsWithStats(combined_mask)
    blob_mask = np.zeros_like(combined_mask)
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
    combined_mask = cv.subtract(combined_mask, blob_mask)

    # Overlay snow mask with original image
    if not GRAY_MODE:
        combined_mask = cv.merge(
            [np.zeros_like(combined_mask), np.zeros_like(combined_mask), combined_mask]
        )
    P_overlay = cv.addWeighted(disp_frame, 1, combined_mask, 0.5, 0)

    # Overlay FPS
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

    # Concatenate the two images horizontally
    concatenated_image = cv.hconcat([P_overlay, combined_mask])

    # Show images
    cv.imshow("Output", concatenated_image)
    cv.imshow("Original", disp_frame)
    if cv.waitKey(1) == ord("q"):
        break

    start_time = time()
    idx += 1
    disp_frame = frame

    # If the last frame is reached, reset the capture and the frame_counter
    if frame_counter == NUM_FRAMES - 1:
        frame_counter = 0  # Or whatever as long as it is the same as next line
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
