import numpy as np
import cv2
import glob

class CameraCalibration:
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        self.obj_points = []  # 3D points in real world space
        self.img_points = []  # 2D points in image plane
        self.calibrated = False
        self.mtx = None  # Camera matrix
        self.dist = None  # Distortion coefficients

    def calibrate(self):
        images = glob.glob('*.jpg')          #has all the images with extension .jpg
        for image in images:
            img = cv2.imread(image)    #reads in each picture from images
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #makes a copy of that particular image

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

            if ret:
                objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0:self.pattern_size[1], 0:self.pattern_size[0]].T.reshape(-1, 2)

                self.obj_points.append(objp)
                self.img_points.append(corners)

        if len(self.obj_points) > 0:
            ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(self.obj_points, self.img_points, gray.shape[::-1], None, None)
            self.calibrated = ret

    def undistort(self, image):
        if self.calibrated:
            return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        else:
            raise Exception("Camera is not calibrated.")
