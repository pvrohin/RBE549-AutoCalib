#Autocalib
import numpy as np
import cv2
import glob
import os
import sys
import time
import math
import matplotlib.pyplot as plt
 
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

#Write a function to calculate the homography matrix without using the cv2.findHomography function 
def get_homography(img_corners,world_corners):
    A = np.zeros((18,9),np.float32)
    for i in range(0,18,2):
        A[i] = [world_corners[int(i/2)][0],world_corners[int(i/2)][1],1,0,0,0,-img_corners[int(i/2)][0]*world_corners[int(i/2)][0],-img_corners[int(i/2)][0]*world_corners[int(i/2)][1],-img_corners[int(i/2)][0]]
        A[i+1] = [0,0,0,world_corners[int(i/2)][0],world_corners[int(i/2)][1],1,-img_corners[int(i/2)][1]*world_corners[int(i/2)][0],-img_corners[int(i/2)][1]*world_corners[int(i/2)][1],-img_corners[int(i/2)][1]]
    U,S,V = np.linalg.svd(A)
    H = V[-1].reshape(3,3)
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





#Write a main function to read the image to be calibrated and the image of the chessboard
def main():
    #Read the image of the chessboard in pdf form

    world_corners = get_world_corners((9,6),21.5)

    print(world_corners)

    # img = cv2.imread('/Users/rohin/Desktop/WPI_Fall_ 2022_Academics/RBE_549/HW1/checkerboardPattern_page-0001.jpg')
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # #Read the image to be calibrated
    # img2 = cv2.imread('/Users/rohin/Desktop/WPI_Fall_ 2022_Academics/RBE_549/HW1/Calibration_Imgs/IMG_20170209_042606.jpg')
    # cv2.imshow('img',img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


   

