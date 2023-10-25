import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

class CameraCalibration:
    def __init__(self, img_dir):
        images = glob.glob("{}/*".format(img_dir))   #has all the images with extension .jpg
        self.objPoints = []                               #3D points in real world space
        self.imgPoints = []                               #2D points in image plane 
        for image in images:
            img = cv2.imread(image)    #reads in each picture from images
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #makes a copy of that particular image
            objp = np.zeros((6*8, 3), np.float32)
            objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
            
            ret, corners = cv2.findChessboardCorners(gray, (6, 8), None)
            if(ret == True):
                self.imgPoints.append(corners)
                self.objPoints.append(objp)

        
    def UndistortedCam(self, imagetoUndisort):
        image = cv2.imread(imagetoUndisort)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objPoints, self.imgPoints, gray.shape[::-1], None, None)
        dst = cv2.undistort(imagetoUndisort, mtx, dist, None, mtx)
        return dst   
    
cam1 = CameraCalibration('/Users/atarodsadatmostafavinia/Documents/opencv project/Fisheye1_1.jpg')    

myUnimange = cam1.UndistortedCam('image_to_undistort.jpg')
cv2.imshow('Here you go', myUnimange)
cv2.waitKey(1000)
cv2.destroyAllWindows()
