import cv2
import matplotlib.pyplot as plt
import numpy as np

class PerspectiveTransformation:
    
    def __init__(self, myframe):
        self.frame = myframe

    def pointingOut(self):
        image = self.frame

        a = (290, 650)   # bottom left
        b = (1082, 650)  # bottom right
        c = (722, 450)   # top right
        d = (580, 450)   # top left
  
        cv2.circle(image, a, 5, (0, 0, 255), -1)
        cv2.circle(image, b, 5, (0, 0, 0), -1)
        cv2.circle(image, c, 5, (0, 255, 0), -1)
        cv2.circle(image, d, 5, (255, 0, 0), -1)

        #Apply Geometical Transformation
        pts1 = np.float32([d, a, b, c])
        pts2 = np.float32([(0, 0), (0, 200), (300, 200), (300, 0)])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        TransIMG = cv2.warpPerspective(image, matrix, (300, 200))


        return TransIMG
    

    
