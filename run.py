import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import glob

def remove_background_and_crop(image_path, target_width=270, target_height=435):
    """
    Remove background and crop image to focus on the main subject (drink cup)
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None
    
    original = img.copy()
    height, width = img.shape[:2]
    
    # Convert to RGB for better processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create mask using multiple methods for better detection  m
    # Method 1: Color-based segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create mask for non-white/transparent backgrounds
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Invert to get non-background areas
    mask1 = cv2.bitwise_not(white_mask)
    
    # Method 2: Edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to connect nearby edges
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Method 3: Threshold-based
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(mask1, edges_dilated)
    combined_mask = cv2.bitwise_or(combined_mask, thresh)
    
    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # If no contours found, use center crop
        return center_crop_and_resize(original, target_width, target_height)
    
    # Find the largest contour (likely the main subject)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add padding
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(width - x, w + 2 * padding)
    h = min(height - y, h + 2 * padding)
    
    # Crop the image
    cropped = original[y:y+h, x:x+w]
    
    # Resize while maintaining aspect ratio
    return resize_with_aspect_ratio(cropped, target_width, target_height)

def center_crop_and_resize(img, target_width, target_height):
    """
    Fallback: Center crop and resize
    """
    height, width = img.shape[:2]
    
    # Calculate crop dimensions to match target aspect ratio
    target_ratio = target_width / target_height
    current_ratio = width / height
    
    if current_ratio > target_ratio:
        # Image is wider, crop width
        new_width = int(height * target_ratio)
        x_offset = (width - new_width) // 2
        cropped = img[:, x_offset:x_offset + new_width]
    else:
        # Image is taller, crop height
        new_height = int(width / target_ratio)
        y_offset = (height - new_height) // 2
        cropped = img[y_offset:y_offset + new_height, :]
    
    return resize_with_aspect_ratio(cropped, target_width, target_height)

def resize_with_aspect_ratio(img, target_width, target_height):
    """
    Resize image while maintaining aspect ratio, pad if necessary
    """
    height, width = img.shape[:2]
    
    # Calculate scaling factor
    scale_w = target_width / width
    scale_h = target_height / height
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Create canvas with target dimensions
    canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    
    # Calculate position to center the image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    # Place resized image on canvas
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    
    return canvas

def process_images(input_folder, output_folder, target_width=270, target_height=435):
    """
    Process all images in input folder
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            # Process image
            result = remove_background_and_crop(image_path, target_width, target_height)
            
            if result is not None:
                # Save processed image
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_folder, f"{name}{ext}")
                
                cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"Saved: {output_path}")
            else:
                print(f"Failed to process: {image_path}")
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    print("Processing complete!")

def process_single_image(input_path, output_path, target_width=270, target_height=435):
    """
    Process a single image
    """
    result = remove_background_and_crop(input_path, target_width, target_height)
    
    if result is not None:
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Processed image saved: {output_path}")
        return True
    else:
        print(f"Failed to process: {input_path}")
        return False

# Example usage
if __name__ == "__main__":
    # Process all images in the specified folder
    input_folder = r"C:\Users\Devils Nerve\Pictures\GreenLine Optimized\2"
    output_folder = r"C:\Users\Devils Nerve\Pictures\GreenLine Optimized\test"
    
    # Process all images
    process_images(input_folder, output_folder)
    
    print("All images processed successfully!")
