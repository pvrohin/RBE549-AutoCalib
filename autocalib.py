#Autocalib
import numpy as np
import cv2
import glob
import os
import sys
import time
import math
import matplotlib.pyplot as plt
import scipy.optimize
 
def get_world_corners(pattern_size,square_size):
    #pattern_size = (9,6)
    #square_size = 21.5 mm
    world_corners = np.zeros((pattern_size[0]*pattern_size[1],2),np.float32)
    world_corners[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
    world_corners *= square_size
    return world_corners

def get_image_corners(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(img_gray,(9,6),None)
    return corners

def get_homography(img_corners, world_corners):
    if len(img_corners) != len(world_corners) or len(img_corners) < 4:
        raise ValueError("Input corners must have the same number of points and at least 4 points.")
    
    A = []
    for i in range(len(img_corners)):
        img_point = img_corners[i]
        world_point = world_corners[i]

        print(img_point)
        print(world_point)
        
        if len(img_point) != 2 or len(world_point) != 2:
            raise ValueError("Each point must be represented as a list of two coordinates (x, y).")
        
        img_x, img_y = img_point
        world_x, world_y = world_point
        
        A.append([-world_x, -world_y, -1, 0, 0, 0, img_x * world_x, img_x * world_y, img_x])
        A.append([0, 0, 0, -world_x, -world_y, -1, img_y * world_x, img_y * world_y, img_y])
    
    A = np.array(A)
    
    _, _, V = np.linalg.svd(A)
    h = V[-1, :]
    
    H = h.reshape((3, 3))
    
    return H

#Calculate camera intrinsic matrix K from the homography matrix H
def get_camera_matrix(H):
    # Normalize the homography matrix H by its last element
    H_normalized = H / H[2, 2]
    
    # Calculate the intrinsic matrix K
    K = np.zeros((3, 3))
    
    K[0, 0] = np.linalg.norm(H_normalized[:, 0])  # Focal length along X-axis
    K[1, 1] = np.linalg.norm(H_normalized[:, 1])  # Focal length along Y-axis
    
    K[0, 1] = (H_normalized[0, 1] * H_normalized[0, 0] + H_normalized[1, 1] * H_normalized[1, 0]) / (np.linalg.norm(H_normalized[:, 0]) * np.linalg.norm(H_normalized[:, 0]))  # Skew
    K[1, 0] = 0.0  # Assuming no skew
    
    K[0, 2] = H_normalized[0, 2]  # X-coordinate of the principal point
    K[1, 2] = H_normalized[1, 2]  # Y-coordinate of the principal point
    
    K[2, 2] = 1.0  # The last element of K is typically set to 1
    
    return K

#Write a function to calculate the intrinsic and extrinsic parameters of the camera
def get_camera_parameters(H,K):
    #Calculate the intrinsic parameters of the camera
    h1 = H[:,0]
    h2 = H[:,1]
    h3 = H[:,2]
    lamda = 1/(np.linalg.norm(np.dot(np.linalg.inv(K),h1)))
    r1 = lamda*np.dot(np.linalg.inv(K),h1)
    r2 = lamda*np.dot(np.linalg.inv(K),h2)
    r3 = np.cross(r1,r2)
    t = lamda*np.dot(np.linalg.inv(K),h3)
    R = np.zeros((3,3),np.float32)
    R[:,0] = r1
    R[:,1] = r2
    R[:,2] = r3
    #Calculate the extrinsic parameters of the camera
    P = np.zeros((3,4),np.float32)
    P[:,:3] = R
    P[:,3] = t
    return P

#Write a function to calculate the reprojection error
def get_reprojection_error(img_corners,world_corners,P):
    img_corners = np.append(img_corners,np.ones((54,1)),axis=1)
    img_corners = np.transpose(img_corners)
    img_corners = np.dot(P,img_corners)
    img_corners = np.transpose(img_corners)
    img_corners = img_corners[:,:2]/img_corners[:,2:]
    error = np.linalg.norm(img_corners-world_corners)
    return error

#Write a function to non-linearly optimize the parameters of the camera
def optimize_parameters(img_corners,world_corners,P):
    #Calculate the initial reprojection error
    error = get_reprojection_error(img_corners,world_corners,P)
    #Initialize the parameters to be optimized
    params = np.zeros((6,1),np.float32)
    params[0] = P[0,0]
    params[1] = P[1,1]
    params[2] = P[0,2]
    params[3] = P[1,2]
    params[4] = P[0,1]
    params[5] = P[0,2]
    #Define the function to be optimized
    def func(params):
        P[0,0] = params[0]
        P[1,1] = params[1]
        P[0,2] = params[2]
        P[1,2] = params[3]
        P[0,1] = params[4]
        P[1,0] = params[5]
        return get_reprojection_error(img_corners,world_corners,P)
    #Optimize the parameters
    params = scipy.optimize.minimize(func,params,method='Nelder-Mead')
    #Update the parameters
    P[0,0] = params[0]
    P[1,1] = params[1]
    P[0,2] = params[2]
    P[1,2] = params[3]
    P[0,1] = params[4]
    P[1,0] = params[5]
    return P

#Write a function to inverse warp the image
def inverse_warp(img,P):
    #Initialize the warped image
    warped_img = np.zeros(img.shape,np.uint8)
    #Inverse warp the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #Calculate the pixel coordinates in the original image
            img_coords = np.dot(np.linalg.inv(P),np.array([i,j,1]))
            img_coords = img_coords/img_coords[2]
            #Check if the pixel coordinates lie within the image
            if img_coords[0]>=0 and img_coords[0]<img.shape[0] and img_coords[1]>=0 and img_coords[1]<img.shape[1]:
                warped_img[i,j] = img[int(img_coords[0]),int(img_coords[1])]
    return warped_img

#Write a function to visualize the results
def visualize_results(img,img_corners,warped_img):
    #Draw the detected corners on the original image
    for i in range(54):
        cv2.circle(img,(int(img_corners[i,0]),int(img_corners[i,1])),3,(0,0,255),-1)
    #Display the original image
    cv2.imshow('img',img)
    #Display the warped image
    cv2.imshow('warped_img',warped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Write a main function to read the image to be calibrated and the image of the chessboard
def main():
    
    img = cv2.imread('/Users/rohin/Desktop/WPI_Fall_ 2022_Academics/RBE_549/HW1/Calibration_Imgs/IMG_20170209_042606.jpg')
    img_corners = get_image_corners(img)
    world_corners = get_world_corners((9,6),21.5)
    H = get_homography(img_corners,world_corners)
    K = get_camera_matrix(H)
    P = get_camera_parameters(H,K)
    P = optimize_parameters(img_corners,world_corners,P)
    warped_img = inverse_warp(img,P)
    visualize_results(img,img_corners,warped_img)


if __name__ == '__main__':
    main()


   

