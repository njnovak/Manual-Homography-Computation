# Part 2 try 2: Combine 23, warp to fit onto 1, warp 4 to fit onto total
import cv2
from numpy import linalg
import numpy as np
import math


def findFeatures(imgraw1, imgraw2, numkps):
    """
    Function to find good keypoint matches between images
    """
    
    # print(imgraw1)
    # print(imgraw2)
    img1 = imgraw1
    img2 = imgraw2



    orb = cv2.SIFT_create(numkps)

    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    flann = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_FLANNBASED)


    """Uncommment below block to see Keypoints!"""
    # gray = cv2.cvtColor(imgraw1, cv2.COLOR_BGR2GRAY)

    # sift = cv2.xfeatures2d.SIFT_create()

    # kp, __ = sift.detectAndCompute(imgraw1, None)

    # sift_image = cv2.drawKeypoints(gray, kp, imgraw1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    matches = flann.knnMatch(descriptors1, descriptors2, 2)

    goodMatches = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            goodMatches.append(m)

    src_pts = 0
    dst_pts = 0
    if len(goodMatches) >= 4:
        dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 2)
        src_pts = np.float32([keypoints2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 2)
    return src_pts, dst_pts


# Homography detection

def getRandom(src_pts, dest_pts, N):
    """
    Get random points in a given range
    """
    if type(src_pts) == type(8):
        src_len = src_pts
    else:
        src_len = len(src_pts)
    
    r = np.random.choice(len(src_pts), N) # Breaking here as src_pts is an integer type and has no len
    src = [src_pts[i] for i in r]
    dest = [dest_pts[i] for i in r]
    return np.asarray(src, dtype=np.float32), np.asarray(dest, dtype=np.float32)

def findHMat(src, dest, N):
    """
    Calculate homography using lecture-provided steps
    """
    # Get Homography matrix
    H_intermediate = []
    for i in range(N):
        # Loop thru number of points to make homography


        x, y = src[i][0], src[i][1]
        xp, yp = dest[i][0], dest[i][1]

        H_intermediate.append([x, y, 1, 0, 0, 0, -x*xp, -xp*y, -xp])
        H_intermediate.append([0, 0, 0, x, y, 1, -yp*x, -yp*y, -yp])
    
    H_intermediate = np.asarray(H_intermediate)
    # Use SVD to solve system of equations, and where H is not all 0's
    __, __, vh = np.linalg.svd(H_intermediate)
    l = vh[-1, :] / vh[-1, -1] # Get last row
    h = l.reshape(3, 3) # Setup 3x3 matrix as last row needed only
    return h

def findHomography(img1_pts, img2_pts):
    """
    RANSAC-inclusive method for finding best homography between 2 images
    """
    max_i = 0
    max_lsrc = []
    max_ldest = []
    for i in range(200):
        # Tuned parameter for number of iterations
        # src_p, dest_p = getRandom(img1_pts, img2_pts, 4)
        index_rand = np.random.randint(0, len(img1_pts), 4)
        pts1_rand = img1_pts[index_rand]
        pts2_rand = img2_pts[index_rand]

        HMat = findHMat(pts1_rand, pts2_rand, 4)
        # Determine the Homography matrix between each image given the current points

        inlines = 0
        lines_src = []
        lines_dest = []
        for point_1, point_2 in zip(img1_pts, img2_pts):
            #For each of the points in the image, compare them with the current homography.

            point_1_flip = (np.append(point_1, 1)).reshape(3,1)
            point_2_hmat = HMat.dot(point_1_flip)
            point_2_hmat = (point_2_hmat / point_2_hmat[2])[:2].reshape(1,2)[0]
            if cv2.norm(point_2-point_2_hmat) < 10: # Threshold tunes to be 10
                # Determine if it qualifies as an inlier
                inlines += 1
                lines_src.append(point_1)
                lines_dest.append(point_2)
        if inlines > max_i:
            # Reset inliers if they are better this iteration
            max_i = inlines
            max_lsrc = lines_src.copy()
            max_lsrc = np.asarray(max_lsrc, dtype=np.float32)
            max_ldest = lines_dest.copy()
            max_ldest = np.asarray(max_ldest, dtype=np.float32)
    # Find final Homography
    Hf = findHMat(max_lsrc, max_ldest, max_i)
    return Hf


img1 = cv2.imread('image_1.jpg')
img2 = cv2.imread('image_2.jpg')
img3 = cv2.imread('image_3.jpg')
img4 = cv2.imread('image_4.jpg')
sift = cv2.SIFT_create(300)



# Warp 3 onto 2
src_pts, dst_pts = findFeatures(img2, img3, 400)


print(len(src_pts))
H = findHomography(src_pts,dst_pts)

dst_23 = cv2.warpPerspective(img3, H, ((img2.shape[1] + img3.shape[1]), img3.shape[0]))

dst_23[0:img2.shape[0], 0:img2.shape[1]] = img2

# cv2.imshow('image 3 warped to 2', dst_23)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Warp 4 onto 3
src_pts, dst_pts = findFeatures(img3, img4, 400)
print(len(src_pts))
H = findHomography(src_pts,dst_pts)
dst_34 = cv2.warpPerspective(img4, H, ((img4.shape[1] + img3.shape[1]), img4.shape[0]))

dst_34[0:img3.shape[0], 0:img3.shape[1]] = img3

# cv2.imshow('image 3 warped to 2', dst_34)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Warp 2 temp images:

tmp_img1 = dst_23
tmp_img2 = dst_34
src_pts, dst_pts = findFeatures(tmp_img1, tmp_img2, 400)
print(len(src_pts))
H = findHomography(src_pts,dst_pts)
dst_tmp = cv2.warpPerspective(tmp_img2, H, ((tmp_img2.shape[1] + tmp_img1.shape[1]), tmp_img2.shape[0]))

for col in range(tmp_img1.shape[0]):
    for row in range(tmp_img1.shape[1]):
        # print(tmp_img1[col][row])
        pix = tmp_img1[col][row]
        if pix[0] != 0 or pix[1] != 0 and pix[2] != 0:
            dst_tmp[col, row] = tmp_img1[col,row]

#Warp final image onto 1:
tmp_img3 = dst_tmp
src_pts, dst_pts = findFeatures(img1, tmp_img3, 400)
print(len(src_pts))
H = findHomography(src_pts,dst_pts)
dst_1234 = cv2.warpPerspective(tmp_img3, H, ((img1.shape[1] + tmp_img3.shape[1]), tmp_img3.shape[0]))
# cv2.imshow("Final warp call", dst_1234)
dst_1234[0:img1.shape[0], 0:img1.shape[1]] = img1

# Display results and wait to close
cv2.imshow('image 2, 3, and 4 warped to 1', dst_1234)
cv2.imwrite("FullStitch.jpg",dst_1234)

# Let the user view it!
cv2.waitKey(0)
cv2.destroyAllWindows()

