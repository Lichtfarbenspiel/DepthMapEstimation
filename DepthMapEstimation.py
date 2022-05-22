# Code adapted from https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
# and https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt


title = "Depth Map Estimation"

def detectKeyPoints(img1, img2):
    # Initiate SIFT detector and find the keypoints and descriptors using SIFT_create
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
 
    imgSift1 = cv2.drawKeypoints(
        img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    imgSift2 = cv2.drawKeypoints(
        img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow(title, np.concatenate((imgSift1, imgSift2), axis=1))
    cv2.waitKey(0)
    return des1, des2, kp1, kp2
    
def matchKeyPoints(img1, img2, des1, des2, kp1, kp2):
    # Match keypoints using a FlannBasedMatcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k = 2)
    

    # Ratio test as per Lowe's paper - only use matches with a reasonable small distance
    pts1 = []
    pts2 = []
    matchesMask = [[0, 0] for i in range(len(matches))]

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    imgMatches = cv2.drawMatchesKnn(
        img1 = img1, keypoints1 = kp1, img2 = img2, keypoints2 = kp2, matches1to2 = matches[300:500], 
        matchesMask = matchesMask[300:500], outImg = None,
        matchColor = (0, 255, 0), singlePointColor = (255, 0, 0), 
        flags = cv2.DrawMatchesFlags_DEFAULT)

    cv2.imshow(title, imgMatches)
    cv2.waitKey(0)

    return pts1, pts2

def computeEpilines(img1, img2, pts1, pts2, ptsToReshape):
        lines = cv2.computeCorrespondEpilines(ptsToReshape.reshape(-1, 1, 2), 2, F)
        lines = lines.reshape(-1, 3)

        r, c = img1.shape

        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        np.random.seed(0) # same random seed for both images

        for r, pt1, pt2 in zip(lines, pts1, pts2,):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0,y0 = map(int, [0, -r[2] / r[1]])
            x1,y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
            img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
            
        return img1, img2

def computeDisparity(img1, img2):
    blockSize = 16
    minDisparity = -32
    maxDisparity = 32
    numDisparities  = maxDisparity - minDisparity

    uniquenessRatio = 5
    speckleWindowSize = 200
    speckleRange = 2
    disp12MaxDiff = 0

    stereo = cv2.StereoSGBM_create(
        minDisparity = minDisparity,
        numDisparities = numDisparities,
        blockSize = blockSize,
        P1 = 8 * 1 * blockSize * blockSize,
        P2 = 32 * 1 * blockSize * blockSize,
        disp12MaxDiff = disp12MaxDiff,
        uniquenessRatio = uniquenessRatio,
        speckleWindowSize = speckleWindowSize,
        speckleRange = speckleRange
    )

    disparity = stereo.compute(img1, img2)
    return disparity

def computeMeanDepth(disparities):
    h, w = disparities[0].shape
    f = 37
    allPoints = []
    depth = np.array(disparities[0].shape, np.float32)

    for disparity in disparities:
        points = np.zeros((h, w, 3), np.float32)
        for v in range(h):
            for u in range(w):
                points[v, u, 0] = u / disparity[v, u]
                points[v, u, 1] = v / disparity[v, u]
                points[v, u, 2] = f / disparity[v, u]

        allPoints.append(points)

    for i in range(1, len(allPoints)):           
        for v1 in range(h): 
            for u1 in range(w):
                nearPointsZ = []
                for v2 in range(h): 
                    for u2 in range(w):
                        if np.linalg.norm(allPoints[0][v1, u1] - allPoints[i][v2, u2]) < 0.1:
                            nearPointsZ.append(allPoints[i][v2, u2, 2])
                            
                depth[v1, u1] = sum(sum(nearPointsZ), allPoints[0][v1, u1, 2]) / (1 + len(nearPointsZ))
    
    return depth

                            

# -------------------------- #


# load images
imageFilenames = glob.glob("./images/IMG_*.jpg")

images = []
disparities = []


for imageFilename in imageFilenames:
    image = cv2.imread(imageFilename, 0)
    image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
    
    images.append(image)

img1 = images[0]

for index in range(len(imageFilenames) - 1):
    index += 1

    img2 = images[index]
    print(index)
        
    cv2.imshow(title, np.concatenate((img1, img2), axis=1))
    cv2.waitKey(0)

    # Detect and draw key points
    des1, des2, kp1, kp2 = detectKeyPoints(img1 = img1, img2 = img2)

    # Match key points and show matched pairs
    pts1, pts2 = matchKeyPoints(img1 = img1, img2 = img2, des1 = des1, des2 = des2, kp1 = kp1, kp2  = kp2)
   
    # Compute the Fundamental Matrix.
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    F, mask = cv2.findFundamentalMat(points1 = pts1, points2 = pts2, method = cv2.FM_RANSAC)

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Compute and draw epilines, return the two images
    img5, img6 = computeEpilines(img1 = img1, img2 = img2, pts1 = pts1, pts2 = pts2, ptsToReshape = pts2)
    img3, img4 = computeEpilines(img1 = img2, img2 = img1, pts1 = pts2, pts2 = pts1, ptsToReshape = pts1)

    cv2.imshow(title, np.concatenate((img3, img5), axis=1))
    cv2.waitKey(0)

    # Stereo rectification
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    _, H1, H2 = cv2.stereoRectifyUncalibrated(points1 = np.float32(pts1), points2 = np.float32(pts2), F = F, imgSize = (w1, h1))

    # rectify both images
    img1_rect = cv2.warpPerspective(src = img1, M = H1, dsize = (w1, h1))
    img2_rect = cv2.warpPerspective(src = img2, M = H2, dsize = (w2, h2))

    cv2.imshow(title, np.concatenate((img1_rect, img2_rect), axis=1))
    cv2.waitKey(0)

    # generate depth map
    disparity = computeDisparity(img1_rect, img2_rect)
    
    disparities.append(disparity)
    print(len(disparities))

    # normalise Values
    disparityDisplay = np.copy(disparity)
    disparityDisplay = cv2.normalize(src = disparityDisplay, dst = disparityDisplay, alpha = 255, beta = 0, norm_type = cv2.NORM_MINMAX)
    disparityDisplay = np.uint8(disparityDisplay)

    # plt.imshow(disparityDisplay, cmap='plasma')
    # plt.colorbar()
    # plt.show()

    cv2.imshow(title, disparityDisplay)
    cv2.waitKey(0)

    print("Image pair", index)

# combine depth maps
meanDepth = computeMeanDepth(disparities)
# normalise Values
meanDepthDisplay = np.zeros(meanDepth.shape, np.uint8)
meanDepthDisplay = cv2.normalize(src = meanDepth, dst = meanDepthDisplay, alpha = 255, beta = 0, norm_type = cv2.NORM_MINMAX)

cv2.imshow(title, meanDepthDisplay)
cv2.waitKey(0)

cv2.destroyAllWindows()