#LANE LINE DETECTION PROJECT
# Project Overview:

This project deals with the challenge of lane line detection using video footage captured from a car's front camera. The main goal is to identify and analyze lane lines in the video. The following steps outline the process:

1. **Camera Calibration and Distortion Coefficients:**
   - Compute camera calibration and distortion coefficients to rectify distortions in raw images.

2. **Optimal Thresholding:**
   - Apply optimal thresholds to enhance lane line visibility in the images.

3. **Perspective Transform:**
   - Transform each frame of the video to achieve a bird's-eye view for improved lane line detection.

4. **Sliding Window Method:**
   - Employ the sliding window method to determine lane curvature and the vehicle's position relative to the lane center.

# How to Use / Requirements:

Before getting started, please ensure that you meet these requirements:

- **Python IDE:** Make sure you have a Python Integrated Development Environment (IDE) installed on your computer.

- **OpenCV:** If OpenCV is not already installed on your system, you can refer to this [link](https://pypi.org/project/opencv-python/) for instructions on setting up OpenCV in Python.

# Libraries and Their Roles:

- **cv2 (OpenCV):** Used for image processing, color space conversions, and various functions.
- **numpy as np:** Optimize calculations within the project.
- **matplotlib.pyplot as plt:** For plotting and displaying critical outputs.
- **matplotlib.image as mpimg:** Handle image reading operations.

# Object-Oriented Programming:

This project follows an object-oriented approach. Several .py files represent important classes necessary to achieve our goals. Detailed explanations of each file's usage and logic are provided in the following section. The project includes a file named 'main.py.' To run the project, follow these steps:

1. Download all project files and place them in a single folder.
2. Open the folder in your preferred Python IDE, such as Visual Studio Code.
3. Execute 'main.py' to initiate the project.
