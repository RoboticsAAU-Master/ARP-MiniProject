import stereo_matching_fast as sm
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

cap_left = cv2.VideoCapture('3D_L4948.MP4')
cap_right = cv2.VideoCapture('3D_R0219.MP4')

video_length_right = cap_right.get(cv2.CAP_PROP_FRAME_COUNT)
video_length_left = cap_left.get(cv2.CAP_PROP_FRAME_COUNT)

min_length = np.min([video_length_left, video_length_right])

fig, ax = plt.subplots(1, 2)
plt.ion()
stereo = cv2.StereoBM_create(16*2, 5)

def pool_expand(arr, kernel_size):
    m, n = arr.shape
    sliced = arr.reshape(m // kernel_size, kernel_size, n // kernel_size, kernel_size)
    pooled = np.mean(sliced, (1, 3))
    expanded_pooled = np.repeat(np.repeat(pooled, kernel_size, axis=0), kernel_size, axis=1)
    return expanded_pooled

def skip_to_time(capture_object, timestamp_seconds):
    # Get the frames per second (fps) of the video
    fps = capture_object.get(cv2.CAP_PROP_FPS)

    # Calculate the frame number based on the specific time
    frame_number = int(timestamp_seconds * fps)

    # Set the capture's position to the calculated frame number
    capture_object.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
skip_to_time(cap_left, 120)
skip_to_time(cap_right, 120)

for f in tqdm(range(int(min_length-1))):
    
    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()
    if ret_l is None or ret_r is None:
        break
    
    frame_gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
    frame_gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
    
    frame_gray_l = cv2.resize(frame_gray_l, (0,0), fx=0.5, fy=0.5)
    frame_gray_r = cv2.resize(frame_gray_r, (0,0), fx=0.5, fy=0.5)
    
    frame_gray_l = sm.preprocess_frame(frame_gray_l)
    frame_gray_r = sm.preprocess_frame(frame_gray_r)
    
    disparity_map = sm.calc_disparity_grad(frame_gray_l, frame_gray_r, num_disparities=16*4, window_size=5)
    disparity_map = pool_expand(disparity_map, 1)
    ax[0].imshow(disparity_map, cmap="gray")
    
    
    disp_cv = stereo.compute(frame_gray_l, frame_gray_r)
    ax[1].imshow(disp_cv, cmap="gray")
    plt.waitforbuttonpress()
    ax[0].clear()
    ax[1].clear()
    
    


