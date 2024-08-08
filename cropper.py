############
# Data Preprocess for locating all the image data at the same sport
# The data images we collected has a bright round area in the center and dark around in background. However, the bright rounds are not in the same spot. This code is decided to crop the images so their pixels match each other for the following process.
# Input: Rare images data has bright round and dark background. Not at the same sport
# Output: Cropped image each image has at the same spot so the pixels match each other
# Author: Tianluke33
# Data: 7/30/2024
############
import cv2
import numpy as np
import pandas as pd
import os

def detect_circle(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image file {image_path} not found.")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise and improve detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Apply Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=30, 
        param1=50, 
        param2=30, 
        minRadius=0, 
        maxRadius=0
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            if i[2] >= 85:
                # Return the first circle with radius >= 85
                return (image_path, i[0], i[1], i[2])
    return None

def calculate_average_center(circle_data):
    total_x = 0
    total_y = 0
    count = 0

    for data in circle_data:
        total_x += data[1]
        total_y += data[2]
        count += 1

    if count == 0:
        return None

    avg_x = total_x // count
    avg_y = total_y // count

    return avg_x, avg_y

def crop_image(image_path, avg_center_x, avg_center_y, output_folder):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image file {image_path} not found.")
        return None

    # Calculate crop coordinates
    left = max(avg_center_x - 100, 0)
    right = min(avg_center_x + 100, image.shape[1])
    top = max(avg_center_y - 100, 0)
    bottom = min(avg_center_y + 100, image.shape[0])

    # Crop the image
    cropped_image = image[top:bottom, left:right]

    # Generate new file name
    filename = os.path.basename(image_path)
    new_filename = os.path.join(output_folder, filename)

    # Save the cropped image
    cv2.imwrite(new_filename, cropped_image)

    return new_filename

def process_images(folder_path, output_folder):
    circle_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            circle = detect_circle(image_path)
            if circle:
                circle_data.append(circle)

    # Calculate average center of all detected circles
    avg_center = calculate_average_center(circle_data)
    if avg_center is None:
        print("No circles detected in the images.")
        return

    # Crop images around the average center and save
    for data in circle_data:
        crop_image(data[0], avg_center[0], avg_center[1], output_folder)

    return circle_data

def save_to_csv(circle_data, output_file):
    df = pd.DataFrame(circle_data, columns=['Image', 'Center_X', 'Center_Y', 'Radius'])
    df.to_csv(output_file, index=False)

# Define folder paths and output file
folder_path = 'FLATS' # Input folder
output_folder = 'ProcessedBackground' # output folder
output_file = 'circle_data.csv'

# Create the output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Process images and save data
circle_data = process_images(folder_path, output_folder)
if circle_data:
    save_to_csv(circle_data, output_file)
    print(f"Processed {len(circle_data)} images, saved cropped images to {output_folder}, and saved circle data to {output_file}.")
else:
    print("No circles were detected or processed.")
