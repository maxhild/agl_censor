import cv2
from east_as_a_function import east_text_detection
from censor import blur_function
from PIL import Image
from transformers_ocr import process_files
import uuid
import os

def main(image_path, east_path='frozen_east_text_detection.pb', min_confidence=0.5, width=320, height=320):
    # Perform text detection
    boxes = east_text_detection(image_path, east_path, min_confidence, width, height)
    #original_image = cv2.imread(image_path)
    #original_height, original_width = original_image.shape[:2]
        
    # Read the image again to apply blurring because east_text_detection does not return the image
    image = cv2.imread(image_path)
    
    # Apply blurring on the detected text areas
    blur_function(image, boxes)
    
    # Optionally save or display the blurred image
    cv2.imwrite('blurred_image.jpg', image)
    print(image_path)
    crop_and_save(image_path, boxes)
    # cv2.imshow('Blurred Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
def crop_and_save(image_path, boxes):
    cropped_image_paths = []
    image_texts = {}  # Dictionary to store image paths and their corresponding texts with unique IDs

    # Ensure the base directory for cropped images exists
    base_dir = os.path.dirname(image_path)
    cropped_dir = os.path.join(base_dir, "cropped_images")
    os.makedirs(cropped_dir, exist_ok=True)

    with Image.open(image_path) as img:
        for idx, (startX, startY, endX, endY) in enumerate(boxes):
            # Crop the image using the box coordinates
            cropped_img = img.crop((startX, startY, endX, endY))
            
            # Extract the file extension and create a new file name with it
            #file_extension = os.path.splitext(image_path)[1]
            cropped_img_path = os.path.join(f"cropped_{idx}_{os.path.basename(image_path)}")
            
            # Ensure the file name is valid
            cropped_img_path = ''.join(c for c in cropped_img_path if c.isalnum() or c in '._-')
            
            # Add the file extension
            #cropped_img_path += file_extension

            cropped_img.save(cropped_img_path)
            cropped_image_paths.append(cropped_img_path)

            image_text = process_files(cropped_img_path)

            # Generate a unique ID for the extracted text
            unique_id = str(uuid.uuid4())

            # Save the extracted text and its unique ID in the dictionary
            image_texts[unique_id] = {'path': cropped_img_path, 'text': image_text}

            # Optionally, print or save the image path, text, and unique ID to a file
            print(f"Cropped image saved to: {cropped_img_path}, text: {image_text}, ID: {unique_id}")

    # Return both the paths and the text with unique IDs
    return cropped_image_paths, image_texts


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True,
                    help="path to input image")
    ap.add_argument("-east", "--east", type=str, required=True,
                    help="path to input EAST text detector")
    ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                    help="minimum probability required to inspect a region")
    ap.add_argument("-w", "--width", type=int, default=320,
                    help="resized image width (should be multiple of 32)")
    ap.add_argument("-e", "--height", type=int, default=320,
                    help="resized image height (should be multiple of 32)")
    args = vars(ap.parse_args())

    # Call the main function with parsed arguments
    main(args["image"], args["east"], args["min_confidence"], args["width"], args["height"])