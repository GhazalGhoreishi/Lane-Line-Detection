import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

class CGT:

    def __init__(self, frame):
        self.frame = frame

    def hlsCVT(self, thresh=(150, 255)):
        img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s = hls[:, :, 2]
        binary = np.zeros_like(s)
        binary[(s > thresh[0]) & (s <= thresh[1])] = 255
        return binary
    
    def pipeline(self, s_thresh=(150, 255), sx_thresh=(10, 100)):
        img = np.copy(self.frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        l_channel = hls[:, :, 1]
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sx/np.max(abs_sx))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 255

        s_binary = self.hlsCVT(s_thresh)
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
        binary_image = np.zeros_like(color_binary)
        binary_image[(color_binary >= 100) & (color_binary <= 255)] = 255
        
        return binary_image

    
# image = cv2.imread('signs_vehicles_xygrad.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 9)
# sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 9)
# mag = np.sqrt(np.square(sobelx) + np.square(sobely))
# scale_factor = np.max(mag)/255 
# mag = (mag/scale_factor).astype(np.uint8)
# binary_output = np.ones_like(mag)
# binary_output[(mag >= 30) & (mag <= 100)] = 1    #here

# # returns the magnitude of the gradient
# def mag_thresh(image, sobel_kernal=3, mag_thresh=(0, 255)):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernal)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernal)
#     mag = np.sqrt((sobelx**2) + (sobely**2))
#     # Rescale to 8-bits
#     scale_factor = np.max(mag)/255 
#     mag = (mag/scale_factor).astype(np.uint8)
#     binary_output = np.zeros_like(mag)
#     binary_output[(mag >= mag_thresh[0]) & (mag <= mag_thresh[1])] = 255
#     return binary_output

# # returns the direction of the gradient
# def dir_thresh(img, sobel_kernal=3, thresh=(0, np.pi/2)):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernal)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernal)
#     absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
#     binary_output = np.zeros_like(absgraddir)
#     binary_output[(absgraddir >= thresh[0]) & (absgraddir <=thresh[1])] = 1
#     return binary_output

    
# cv2.imshow('f', mag_thresh(image, 9, (30, 100)))
# # cv2.imshow('f', binary_output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
