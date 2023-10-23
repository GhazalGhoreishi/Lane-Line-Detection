import cv2
import matplotlib.pyplot as plt
import numpy as np

class PerspectiveTransformation:
    
    def __init__(self, myframe):
        self.frame = myframe

    def pointingOut(self):
        image = self.frame

        a = (160, 650)   # bottom left
        b = (1100, 650)  # bottom right
        c = (750, 450)   # top right
        d = (560, 450)   # top left
  
        cv2.circle(image, a, 5, (0, 0, 255), -1)
        cv2.circle(image, b, 5, (0, 0, 0), -1)
        cv2.circle(image, c, 5, (0, 255, 0), -1)
        cv2.circle(image, d, 5, (255, 0, 0), -1)

        #Apply Geometical Transformation
        pts1 = np.float32([d, a, b, c])
        pts2 = np.float32([(0, 0), (0, 720), (1280, 720), (1280, 0)])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        TransIMG = cv2.warpPerspective(image, matrix, (1280, 720))


        return TransIMG
    

    