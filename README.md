Drink Cup Image Processor
Python script to remove backgrounds and crop images of drink cups to a specified size (default: 270x435 pixels).
Features

Removes white/transparent backgrounds
Detects main subject using color segmentation, edge detection, and thresholding
Crops to focus on the drink cup
Resizes while maintaining aspect ratio
Processes single images or entire folders
Supports multiple image formats (JPG, PNG, BMP, TIFF, WEBP)

Requirements

Python 3.x
OpenCV (cv2)
NumPy
Pillow (PIL)

Install dependencies:
pip install opencv-python numpy pillow

Usage
Process a folder of images
from script import process_images

input_folder = "path/to/input/folder"
output_folder = "path/to/output/folder"
process_images(input_folder, output_folder, target_width=270, target_height=435)

Process a single image
from script import process_single_image

input_path = "path/to/image.jpg"
output_path = "path/to/output.jpg"
process_single_image(input_path, output_path, target_width=270, target_height=435)

How It Works

Background Removal: Combines color-based segmentation, edge detection, and thresholding to create a mask.
Subject Detection: Finds the largest contour (assumed to be the drink cup).
Cropping: Crops around the subject with padding.
Resizing: Resizes to target dimensions while preserving aspect ratio, padding with white if needed.
Fallback: If no subject is detected, performs a center crop.

Limitations

Optimized for drink cup images with clear backgrounds
May struggle with complex backgrounds or low-contrast images
Assumes the largest contour is the main subject

License
MIT License
