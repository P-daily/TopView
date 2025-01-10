import cv2
import numpy as np
from matplotlib import pyplot as plt


def crop_image_based_on_green_markers(frame):
    """
    Crops a frame based on the positions of green markers.

    Parameters:
        frame (numpy.ndarray): A single frame from a video or image.

    Returns:
        cropped_image (numpy.ndarray): Cropped image based on green markers.
        green_centers (list): List of detected green marker centers.
    """
    # Convert the frame to RGB and HSV color spaces
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV
    lower_green = np.array([40, 50, 50])  # Lower bound of green in HSV
    upper_green = np.array([80, 255, 255])  # Upper bound of green in HSV

    # Create a mask for green color
    green_mask = cv2.inRange(frame_hsv, lower_green, upper_green)

    # Find contours of the green regions
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store centers of the green squares
    green_centers = []

    # Loop through the contours to find the centers
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:  # Avoid division by zero
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            green_centers.append((cx, cy))

    # Sort the centers to ensure consistent order (top-left to bottom-right)
    green_centers = sorted(green_centers, key=lambda x: (x[1], x[0]))

    # Check if exactly 4 green markers are found
    if len(green_centers) != 4:
        raise ValueError(f"Expected 4 green markers, but found {len(green_centers)}")

    # Define the bounding box based on the detected green centers
    x_min = min([point[0] for point in green_centers])
    x_max = max([point[0] for point in green_centers])
    y_min = min([point[1] for point in green_centers])
    y_max = max([point[1] for point in green_centers])

    # Crop the frame using the bounding box
    cropped_image = frame_rgb[y_min:y_max, x_min:x_max]

    # Return the cropped image and the green centers
    return cropped_image, green_centers
