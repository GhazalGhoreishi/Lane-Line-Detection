import numpy as np
import cv2
import matplotlib.pyplot as plt

class LineDetection:
    
    def __init__(self, binary_warped):
        self.frame = binary_warped
        self.out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    def histogram(self):
        binary_warped = self.frame
        hist = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        return hist 

    def base(self):   # Returns leftx_base and rightx_base
        histogram = self.histogram()
        midpoint = int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint 
        return leftx_base, rightx_base
    
    def find_lane_pixels(self):
        binary_warped = self.frame
        histogram = self.histogram()
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        leftx_base, rightx_base =  self.base()

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 10
        # Set minimum number of pixels found to recenter window
        minpix = 10

        # Set height of windows - based on nwindows above and image shape
        window_height = int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # # Draw the windows on the visualization image
            # cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            # (win_xleft_high,win_y_high),(0,255,0), 2) 
            # cv2.rectangle(out_img,(win_xright_low,win_y_low),
            # (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty


    def fit_polynomial(self):
        ym_per_pix = 30/720
        xm_per_pix = 3.7/700
        binary_warped = self.frame
        leftx, lefty, rightx, righty = self.find_lane_pixels()

        left_fit = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

        # left_fit = np.polyfit(lefty, leftx, 2)
        # right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        image = self.frame
        output_image = np.dstack((image, image, image))

        points1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        points2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        points = np.hstack((points1, points2))
        cv2.fillPoly(output_image, np.int_([points]), color = [0, 127, 0])
        
        output_image[lefty, leftx] = [255, 0, 0]
        output_image[righty, rightx] = [0, 0, 255]


        
        return left_fitx, right_fitx, ploty, output_image
    
    def measure_curvature_pixels(self):
        left_fitx, right_fitx, ploty, img = self.fit_polynomial()
        left_fit = np.polyfit(ploty, left_fitx, 2)
        right_fit = np.polyfit(ploty, right_fitx, 2)
        
    
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        
        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        
        return left_curverad, right_curverad

    def CenDetect(self):
        left_fitx, right_fitx, ploty, img = self.fit_polynomial()
        xm_per_pix = 3.7/700
        lane_center =  (right_fitx[20] + left_fitx[20]) / 2 + (xm_per_pix * 540) # Lane center in meters
        car_position = 640 * xm_per_pix # Middle of the car

        # Calculate off-center distance
        off_center_distance = car_position - lane_center

        # Format the result as a string
        if off_center_distance > 0:
            off_center_str = f"{abs(off_center_distance):.3f}m right of center"
        elif off_center_distance < 0:
            off_center_str = f"{abs(off_center_distance):.3f}m left of center"
        else:
            off_center_str = "Centered"

        return off_center_str