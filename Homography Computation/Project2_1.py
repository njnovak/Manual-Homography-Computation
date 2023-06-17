
# Part 1:
import cv2
import matplotlib
import numpy as np
from numpy import linalg
import math
import matplotlib
from sympy import N
import matplotlib.pyplot as plt
"""Requirements 
Design an image processing pipeline to extract the paper on the ground and then extract all
of its corners using the Hough Transformation technique .

Once you have all the corner points, you will have to compute homography between real
world points and pixel coordinates of the corners. You must write your own function to
compute homography.

Decompose the obtained homography matrix to get the rotation and translation"""



def hough_line_3(img, angle_step=1, lines_are_white=True, value_threshold=5):
    
    rho_resolution = 1
    theta_resolution = 1
    height, width = img.shape # we need heigth and width to calculate the diag line
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2)) # pythagorean theorem
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution) # Set up rhos array
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution)) # Set up thetas array

    # create the empty Hough Accumulator with dimensions equal to the size of rhos and thetas
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)
    y_idxs, x_idxs = np.nonzero(img) # find all edge (nonzero) pixel indexes

    for i in range(len(x_idxs)): # cycle through edge points
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)): # cycle through thetas and calc rho
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1

    return H, rhos, thetas

def hough_lines_acc(img, rho_resolution=1, theta_resolution=1):
    """
    Function to build the accumulator for a given image of edges
    """
    height, width = img.shape # we need heigth and width to calculate the diagonal line
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2)) # Pythag theorem
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))

    # create the empty Hough Accumulator with dimensions equal to the size of rhos and thetas
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img) # find all edge pixels

    for i in range(len(x_idxs)): # loop these edges points
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)): # loop thetas and calc rho with polar line formula
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1

    return H, rhos, thetas


def hough_peaks2(H, num_peaks, threshold=0, nhood_size=3):
    """
    Uses a neighborhood search to return [num_peaks] local maxima in an accumulator array
    """
    # loop number of peaks to identify them
    indicies = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1) # find argmax in flattened array
        H1_idx = np.unravel_index(idx, H1.shape) # remap to shape of Hough
        indicies.append(H1_idx)

        # surpess indicies in neighborhood
        idx_y, idx_x = H1_idx # first separate x, y indexes from argmax(H)

        # If x is near the low edge:
        if (idx_x - (nhood_size/2)) < 0: min_x = 0
        else: min_x = idx_x - (nhood_size/2)

        # If x is near the high edge:
        if ((idx_x + (nhood_size/2) + 1) > H.shape[1]): max_x = H.shape[1]
        else: max_x = idx_x + (nhood_size/2) + 1


        # IF y is near the low edge:
        if (idx_y - (nhood_size/2)) < 0: min_y = 0
        else: min_y = idx_y - (nhood_size/2)

        # If y is near the high edge:
        if ((idx_y + (nhood_size/2) + 1) > H.shape[0]): max_y = H.shape[0]
        else: max_y = idx_y + (nhood_size/2) + 1

        # Bound each index by the neighborhood size and set all values to 0 for next loop
        # This way, nothing in this area will be double-selected.
        # Many frames have very bright regions so this avoids just choosing 4 points in a tight region
        for x in range(int(min_x), int(round(max_x))):
            for y in range(int(min_y), int(round(max_y))):
                
                # remove neighbors and point in H1
                H1[y, x] = 0

                # highlight peaks in original Hough for ease of next loop
                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255

    # return the indicies and the original Hough space with selected points
    return indicies, H


def draw_hough_lines(src_img, indicies, rhos, thetas):
    """
    Draw lines perpendicular to the a line drawn from the
    x and y points corresponding to rho and theta to another scalar multiple of those x and y values.

    Calculate the slope of these lines, their y intercept, and two points on them. Map them to a dictionary index
    for the next step and return the dict.
    """
    lines = {} # Will contain vital line information for future steps
    for i in range(len(indicies)):
        # Exctract line information from polar values
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        
        # Get line a and b values (intersect point (a,b))
        a = np.cos(theta)
        b = np.sin(theta)
        
        # Set second point on the line, scalar m ultiple of the first
        x0 = a*rho
        y0 = b*rho

        # Draw lines across the whole image so I can see what they look like and tune accordingly
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        # Draw a line on the image with the obtained normal line, representing and edge
        cv2.line(src_img, (x1, y1), (x2, y2), (200, 190, 0), 2)

        # Cache its parameters for later
        slope = (y2-y1)/(x2-x1)
        b = y1 - (slope * x1)
        lines[i] = [slope,b,(x1,y1),(x2,y2)]
    return lines

# Found and adapted this handy function from https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
        px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
        py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
        return [px, py]

def get_cross_points(lines):
    """
    Function to find the intersection of two lines and determine if it is a corner
    """
    angles = []
    best_angles = []
    intersections = []

    for i in range(len(lines)):
        m1 = lines[i][0]
        for j in range(len(lines) - 1):
            if i != j:
                m2 = lines[j][0]
                theta = math.degrees(math.atan((m2-m1) / (1+(m1*m2))))
                angles.append([theta,i,j])
                if theta < 120 and theta > 30:
                    # If intersection angle is close to 90
                    point1 = lines[i][2]
                    point2 = lines[i][3]
                    point3 = lines[j][2]
                    point4 = lines[j][3]
                    x_0, y_0=findIntersection(point1[0],point1[1],point2[0],point2[1],point3[0],point3[1],point4[0],point4[1])
                    intersections.append((x_0, y_0))
                    # Cache the good intersection for later use

    # print("intersections: ",intersections)
    
    return intersections

def cam_frame_clockwise(points):
    """
    Point categorization for comparison to the world points of the paper. Kept naming same for both
    """
    # Origin is top left
    min_vert = (0,2000000)
    max_vert = (0,-2000000)
    min_horiz = (2000000,0)
    max_horiz = (-2000000,0)
    for i in points:
        x = i[0]
        y = i[1]
        if x > max_horiz[1]:
            max_horiz = i
        if x < min_horiz[1]:
            min_horiz = i
        if y > max_vert[0]:
            max_vert = i
        if y < min_vert[0]:
            min_vert = i
    return[min_horiz, max_vert, max_horiz, min_vert]

def world_points():
    """
    Point categorization for paper at origin. Same naming and convention as camera frame for ease of matching later
    """
    #21.6 cm x 27.9 cm.
    #Origin at top left, 0,0
    #Paper aligned horizontally
    # Pairs in x,y
    max_vert = (0,0)
    min_horiz = (0,-21.6)
    min_vert = (27.9, -21.6)
    max_horiz = (27.9,0)
    return[min_horiz, max_vert, max_horiz, min_vert]

    


def get_K_Mat(scale_factor):
    """
    Gets the given K matrix of the camera to map from cm to pixels or pixels to cm
    """
    new_points = []
    
    focal = 1.38e3 * scale_factor
    s = 0 * scale_factor
    px = 9.46e2 * scale_factor
    mf = 1.38e3 * scale_factor
    py = 5.27e2 * scale_factor

    K_Mat = np.array([[focal, s, px],[0, mf, py], [0, 0, 1]])

    return K_Mat

def findHMat(src, dest, N):
    '''
    Ported over from part 2, where this was originally used. The same method applies here, though
    '''
    A = []
    for i in range(N):
        x, y = src[i][0], src[i][1]
        xp, yp = dest[i][0], dest[i][1]

        A.append([x, y, 1, 0, 0, 0, -x*xp, -xp*y, -xp])
        A.append([0, 0, 0, x, y, 1, -yp*x, -yp*y, -yp])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H

def get_rotation_values_deg(A):
    """
    yaw=atan2(R(2,1),R(1,1));
    pitch=atan2(-R(3,1),sqrt(R(3,2)^2+R(3,3)^2)));
    roll=atan2(R(3,2),R(3,3));
    from https://stackoverflow.com/questions/11514063/extract-yaw-pitch-and-roll-from-a-rotationmatrix
    """
    yaw = math.atan2(A[1,0], A[0,0])
    pitch = math.atan2(-A[2,0],math.sqrt(A[2,1]**2 + A[2,2]**2))
    roll = math.atan2(A[2,1],A[2,2])
    return roll*180/math.pi, pitch*180/math.pi, yaw*180/math.pi


# Main loop
video_source = cv2.VideoCapture('project2.avi')

if video_source.isOpened() == False:
    print("Error opening video")
mingray = 210
maxgray = 255

min_BGR = np.array([mingray])
max_BGR = np.array([maxgray])
min_HSV = np.array([90,34,233])
max_HSV = np.array([106,76,255])
reading, frame = video_source.read()

# x,y,__ = frame.shape
# cv2.resize(frame,cv2.size(x,y*3))
x,y,__ = frame.shape

xf = int(round(x*0.6))
yf = int(round(y*0.6))
out = cv2.VideoWriter('HoughTransform.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (yf,xf))
frame_counter = 0
while reading:

    # Loop over every frame
    # print(type(x),y)
    reading, frame = video_source.read()

    if reading:

        scale_percent = 0.6 # percent of original size
        fy = frame.shape[1]
        fx = frame.shape[0]
        dim = (int(fy*scale_percent), int(fx*scale_percent))
        shapes = cv2.resize(frame, dim)
        
        shapes_grayscale = cv2.cvtColor(shapes, cv2.COLOR_RGB2GRAY)


        shapes_blurred = cv2.GaussianBlur(shapes_grayscale, (5, 5), 1.5)

        # This took much debugging to realize this needed different thresholds
        edges = cv2.Canny(shapes_blurred, 100, 200)

        hough_mat, rhos, thetas = hough_lines_acc(edges)
        # cv2.imshow("Accumulator", hough_mat)
        # print(hough_mat.shape)

        point1_thetas = [9,53]
        point1_rhos = [1244,1396]
        point2_thetas = [13,53]
        point2_rhos = [1432,1548]
        point3_thetas = [106,143]
        point3_rhos = [1849,1876]
        point4_thetas = [89,149]
        point4_rhos = [1998,2088]


        local_point1_arr = hough_mat[point1_rhos[0] : point1_rhos[1], point1_thetas[0] : point1_thetas[1]]
        local_point2_arr = hough_mat[point2_rhos[0] : point2_rhos[1], point2_thetas[0] : point2_thetas[1]]
        local_point3_arr = hough_mat[point3_rhos[0] : point3_rhos[1], point3_thetas[0] : point3_thetas[1]]
        local_point4_arr = hough_mat[point4_rhos[0] : point4_rhos[1], point4_thetas[0] : point4_thetas[1]]

        H_indicies = [local_point1_arr,local_point2_arr,local_point3_arr,local_point4_arr]

        indicies = [0,0,0,0]
        indicies, H = hough_peaks2(hough_mat, 4, nhood_size = 30)
        # print(indicies)
        lines = draw_hough_lines(shapes, indicies, rhos, thetas)
        intersections = get_cross_points(lines)
        dst_pts = world_points()
        src_pts = cam_frame_clockwise(intersections)

        Homog = findHMat(src_pts,dst_pts,4)
        K_Mat = get_K_Mat(scale_factor=scale_percent)
        E_Mat = np.linalg.inv(K_Mat) * Homog

        
        A1 = E_Mat[:,0]
        A2 = E_Mat[:,0]
        lambda1 = np.linalg.norm(A1)
        lambda2 = np.linalg.norm(A2)

        lambda_E = (lambda2 + lambda1)/2



        # print(lambda1, lambda2)

        rotation1 = E_Mat[:,0]/lambda_E
        rotation2 = E_Mat[:,1]/lambda_E
        translation = E_Mat[:,2]
        rotation3 = np.cross(rotation1, rotation2)

        rot_mat = np.array([rotation1, rotation2, rotation3])
        rot_mat = np.transpose(rot_mat)

        """Uncomment to view useful matrices"""
        # print("Homography: ", Homog)
        # print("E_matrix, divided by lambda", E_Mat/lambda_E)
        
        # print("Rot",rotation1)
        # print(rotation2)
        # print(rotation3)
        roll, pitch, yaw = get_rotation_values_deg(rot_mat)
        # print(roll, pitch, yaw)
        
        # print(lines)
        text = ""
        text += "Roll (deg): " + str(roll)
        text += " "
        text += "Pitch (deg): " + str(pitch)
        text += " "
        text += "Yaw (deg): " + str(yaw)
        

        text2 = ""
        text2 += "X (cm): " + str(int(translation[0]))
        text += " "
        text2 += "Y (cm): " + str(int(translation[1]))
        text += " "
        text2 += "Z (cm): " + str(int(translation[2]))

        
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (50, 50)
        org2 = (50, 150)
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (255, 0, 255)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.putText() method
        shapes = cv2.putText(shapes, text, org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
        shapes = cv2.putText(shapes, text2, org2, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow("Results", shapes)
        
        key = cv2.waitKey(300) #pauses for .3 seconds before fetching next image
        if key == 27: #if ESC is pressed, exit loop and kill program
            cv2.destroyAllWindows()
            break
        out.write(shapes) # Create the video with information written on it.
    else:
        break
out.release()
# print(frame_counter)

cv2.destroyAllWindows()
