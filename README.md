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


# Classes
As mentioned each file in this project represents a class. Here is a brief review of how they actually work.
# Camera Calibration Class Documentation

## Overview

The `CameraCalibration` class provides functionality for camera calibration using a series of chessboard images. It enables the calibration of a camera, calculation of the camera matrix, and undistortion of images.

## Class Attributes

- `pattern_size` (tuple): The size of the chessboard pattern in the format (number of columns, number of rows).
- `obj_points` (list): A list of 3D points in real-world space.
- `img_points` (list): A list of 2D points in the image plane.
- `calibrated` (bool): A flag indicating whether the camera has been successfully calibrated.
- `mtx` (numpy.ndarray): The camera matrix.
- `dist` (numpy.ndarray): The distortion coefficients.

## Class Methods

1. `__init__(self, pattern_size)`
   - **Initialization Method**
   - Initializes a `CameraCalibration` object with the specified chessboard pattern size.

2. `calibrate(self)`
   - **Calibration Method**
   - Calibrates the camera by finding chessboard corners in a series of images and calculating the camera matrix and distortion coefficients.

3. `undistort(self, image)`
   - **Undistortion Method**
   - Undistorts an input image using the camera matrix and distortion coefficients if the camera is calibrated.

## Usage Example

1. Initialize the `CameraCalibration` object with the desired chessboard pattern size.
   
   ```python
   calibration = CameraCalibration((9, 6))

# Perspective Transformation Class Documentation

## Overview

The `PerspectiveTransformation` class provides functionality for performing a perspective transformation on an input image. This transformation is used to obtain a specific view of the image by changing the perspective and orientation of objects within it.

## Class Attributes

- `frame`: The input image on which the perspective transformation will be applied.

## Class Methods

1. `__init__(self, myframe)`
    - **Initialization Method**
    - Initializes a `PerspectiveTransformation` object with the provided input image.

2. `pointingOut(self)`
    - **Pointing Out Method**
    - Applies perspective transformation to the input image to point out specific regions of interest by defining four key points.

    **Returns:**
    - `TransIMG` (numpy.ndarray): The transformed image with the defined region of interest.

**Usage Example**

1. Initialize the `PerspectiveTransformation` object with the input image:

   ```python
   my_frame = cv2.imread('input_image.jpg')
   perspective_transform = PerspectiveTransformation(my_frame)

# Color and Gradient Thresholding Class Documentation

## Overview

The `CGT` class provides methods for performing color and gradient thresholding on an input image. This class is used for extracting specific features from an image that can aid in various computer vision and image processing tasks.

## Class Attributes

- `frame`: The input image on which the color and gradient thresholding operations will be applied.

## Class Methods

1. `__init__(self, frame)`
   - **Initialization Method**
   - Initializes a `CGT` object with the provided input image.

2. `hlsCVT(self, thresh=(150, 255))`
   - **HLS Color Thresholding Method**
   - Converts the input image to the HLS color space and applies a threshold to the S-channel to extract specific color information.

   **Parameters:**
   - `thresh` (tuple): A tuple specifying the lower and upper thresholds for the S-channel.

   **Returns:**
   - `binary` (numpy.ndarray): A binary image where pixels within the specified threshold range are set to 255.

3. `pipeline(self, s_thresh=(150, 255), sx_thresh=(10, 100))`
   - **Combined Color and Gradient Thresholding Method**
   - Applies a combination of color and gradient thresholding to the input image. The method extracts features from the S-channel of the HLS color space and the gradient of the L-channel.

   **Parameters:**
   - `s_thresh` (tuple): A tuple specifying the lower and upper thresholds for the S-channel.
   - `sx_thresh` (tuple): A tuple specifying the lower and upper thresholds for the gradient in the L-channel.

   **Returns:**
   - `binary_image` (numpy.ndarray): A binary image where pixels within the specified threshold ranges are set to 255, indicating features of interest.

## Usage Example

1. Initialize the `CGT` object with the input image:

   ```python
   input_image = cv2.imread('input_image.jpg')
   color_gradient_thresholding = CGT(input_image)
   

