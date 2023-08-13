import cv2
import numpy as np

def cartoonize_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a median blur to reduce noise
    blurred = cv2.medianBlur(gray, 5)

    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # Apply a bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(image, 9, 75, 75)

    # Combine the filtered image with the edges
    cartoon = cv2.bitwise_and(filtered, filtered, mask=edges)

    return cartoon

# Path to the input image
input_image_path = 'input.jpg'

# Cartoonize the image
cartoon_image = cartoonize_image(input_image_path)

# Display the cartoonized image
cv2.imshow('Cartoonized Image', cartoon_image)
cv2.waitKey(0)
cv2.destroyAllWindows()