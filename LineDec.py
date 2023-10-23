import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class LineDetection:

    def __init__(self, myframe):
        self.frame = myframe  #initializig each frame to the object frame attribute

    def ld(self):
        image = self.frame
        image_copy = image.copy()
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

        blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur_gray, 10, 100)

        mask = np.zeros_like(edges)
        ignore_mask_color = 255


        vertices = np.array([[(160, 650), (1100, 650), (750, 450), (560, 450)]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)

        rho = 1
        theta = np.pi / 180
        threshold = 1
        min_line_lenght = 10
        max_line_gap = 15
        line_image = np.copy(image)*0

        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_lenght, max_line_gap)

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(masked_edges, (x1, y1), (x2, y2), (255, 255, 255), 10)

        color_edges = np.dstack((masked_edges, masked_edges, masked_edges))

        combo = cv2.addWeighted(color_edges, 0.8, image, 1, 0)
        return masked_edges

