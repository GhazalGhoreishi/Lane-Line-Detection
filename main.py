import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from CameraCal import CameraCalibration
from LineDec import LineDetection
from PresTrans import PerspectiveTransformation
from ColGradThresholding import CGT


# Open a video file
cap = cv2.VideoCapture('project_video.mp4')

while cap.isOpened():    #operating each time on every frame of the video
    ret, frame = cap.read()

    if not ret:
        break

    # cam1 = CameraCalibration()  It can be uncommented when you have several checker board images in the same folder.
    # f = cam1.undistort(frame)
    quadratic_coeff = 3e-4 
    f2 = CGT(frame)
    f3 = f2.hlsCVT()
    f4 = PerspectiveTransformation(f3)
    f5 = f4.pointingOut()
    binWarp = LineDetection(f5)

    lc, rc = binWarp.measure_curvature_pixels()
    curve = ((lc + rc)/ 2)
    curve = round(curve, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    text = "Curve radius: " + str(curve) + "m"
    cv2.putText(frame,  
                text,  
                (50, 50),  
                font, 1,  
                (0, 255, 0),  
                2,  
                cv2.LINE_4)
    cv2.putText(frame,  
                binWarp.CenDetect(),  
                (50, 100),  
                font, 1,  
                (0, 255, 0),  
                2,  
                cv2.LINE_4)
    
    leftx_base, rightx_base = binWarp.base()
    lxb = (leftx_base / 200 ) * 1280
    rxb = (rightx_base / 200 ) * 1280
    a = (lxb, 720)
    b = (rxb, 720)

    # left_fitx, right_fitx, ploty, output_image = .fit_polynomial()

    lines = np.zeros_like(frame)
    for i in range(frame.shape[0]):
        if(0 <= int())

    cv2.imshow('video', frame)
    cv2.waitKey(100)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()