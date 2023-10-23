import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from CameraCal import CameraCalibration
from LineDec import LineDetection
from PresTrans import PerspectiveTransformation
from test import LineDec


# Open a video file
cap = cv2.VideoCapture('project_video.mp4')

while cap.isOpened():    #operating each time on every frame of the video
    ret, frame = cap.read()

    if not ret:
        break

    f = LineDetection(frame)
    f1 = f.ld()
    f2 = PerspectiveTransformation(f1)
    f3 = f2.pointingOut()
    # cv2.imshow('Lane Lines Detection', f3)
    # #cv2.imshow('main video', frame)
    ff3 = mpimg.imread(f3)
    ld = LineDec(ff3)
    f4 = ld.histogram()
    cv2.imshow('What can I say?', f4)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()