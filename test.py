import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

class LineDec:
    def __init__(self, frame):
        self.frame = frame

    def histogram(self):
        img = self.frame
        histo = np.sum(img[img[0]//2:,:], axis=0)
        return histo    

