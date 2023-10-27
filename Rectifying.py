import cv2
import matplotlib.pyplot as plt
import numpy as np

def rectify(left, right):
    # Read the images (rectilinear stereo rig)
    #left = cv2.imread('Exercise 2/sword1/im0.png', cv2.IMREAD_GRAYSCALE)
    #right = cv2.imread('Exercise 2/sword1/im1.png', cv2.IMREAD_GRAYSCALE)

    # Prepare the intrinsic parameters of the two cameras
    # we assume no lens distortion, i.e. dist coeffs = 0
    cameraMatrix1 = np.array([[6872.874, 0, 1605.291],
                            [0, 6872.874, 938.212],
                            [0, 0, 1]])
    distCoeffs1 = np.array([0,0,0,0,0])

    cameraMatrix2 = np.array([[6872.874, 0, 1922.709],
                            [0, 6872.874, 938.212],
                            [0, 0, 1]])
    distCoeffs2 = np.array([0,0,0,0,0])

    # Prepare the extrinsic parameters
    rotationMatrix = np.eye(3) # we assume no rotation
    transVector = np.array([-174.724, 0.0, 0.0])

    # Rectify the images using both the intrinsic and extrinsic parameters
    # Note: the image pair is already rectified so we do not really need
    # to do this but we need the Q matrix to compute depth later on
    image_size = left.shape[::-1]
    print(image_size)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1,
                                                    cameraMatrix2, distCoeffs2,
                                                    (image_size[1],image_size[2]), rotationMatrix, transVector)

    # Remap the left image based on the resulting rotation R1 and projection matrix P1
    leftmapX, leftmapY = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (image_size[1],image_size[2]), cv2.CV_32FC1)
    left_remap = cv2.remap(left, leftmapX, leftmapY, cv2.INTER_LANCZOS4)

    # Do the same for the right image
    rightmapX, rightmapY = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (image_size[1],image_size[2]), cv2.CV_32FC1)
    right_remap = cv2.remap(right, leftmapX, rightmapY, cv2.INTER_LANCZOS4)
    
    return left_remap, right_remap

    # Let's try to compare the two images before and after rectifying them
    # Plot the two original images side-by-side
    ax = plt.subplot(211)
    ax.set_title('original')
    ax.imshow(np.hstack([left,right]),'gray')

    # Plot the two remapped images side-by-side
    ax = plt.subplot(212)
    ax.set_title('remapped')
    ax.imshow(np.hstack([left_remap,right_remap]),'gray')
    plt.show()
    # They look pretty much identical with the expection of some extra padding
    # for the rectifying images. You can use both pair of images for the rest
    # of the exercise. It is up to you - both should give valid results.


    # You do the rest! Start by calculating the disparity map

    # Compute the disparity map using the"StereoBM" class. Try to choose sensible values for both the block size and number of disparities. 
    # (you can try to inspect foreground objects to make an estimated guess about the number of disparities)
    # Calculate disparity map
    stereo = cv2.StereoBM.create(numDisparities=256, blockSize=21)
    disparity = stereo.compute(left_remap, right_remap)

    # Plot the disparity map
    plt.imshow(disparity, cmap='gray')
    plt.title('Disparity map')
    plt.show()

    # Reproject the disparity map to 3D using OpenCV's "reprojectImageTo3D(...)"-function. 
    # You will need the Q matrix from the "stereoRectify(...)"-function estimated earlier in the script.
    img_3d = cv2.reprojectImageTo3D(disparity, Q) # z-value corresponds to depth in mm 

    # Try to extract the z-values of the resulting 3D points and plot it as a depth map. (hint: you may have to clip the value to a reasonable range 
    # to avoid noise. Depth values between 0 and 2000 seems reasonable). Does it make sense? How far away is the potted plant from the camera?
    # - The plant is about 50-60 cm away from the camera
    depth_map = img_3d[:,:,2].astype(np.float32)
    depth_map = np.clip(depth_map, 0.0, 2000.0)

    depth_map2 = img_3d[:,:,2].astype(np.float32)
    depth_max = 2000.0
    depth_map2[depth_map2 > depth_max] = depth_max
    depth_map2[depth_map2 < 0.0] = 0.0

    plt.imshow(depth_map, cmap='jet') # The jet parameter makes it colored
    plt.title('Depth map')
    plt.show()

    # The translation between the two cameras is currently set to -174.724 mm - what happens if you change it to meters instead? 
    # Plot the resulting depth map.
    # - The depth map just thinks all pixels are incredibly far away, which is because we have relatively small disparities compared
    # - to the massive baseline. For a small disparity it would therefore suggest that the objects are very far away
