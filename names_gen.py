import cv2
import numpy as np
import random
import gender_guesser.detector as gender
import time
import uuid

# Load the names from the files
with open('names_dict/first_and_last_name_female.txt', 'r') as file:
    female_names = [line.strip() for line in file]

with open('names_dict/first_and_last_name_male.txt', 'r') as file:
    male_names = [line.strip() for line in file]

def add_name_to_image(name, image_path, box, gender_par):
    image = cv2.imread(image_path)
    (startX, startY, endX) = box

    # Set the text position (top-left corner of the bounding box)
    text_position = (startX, startY - 10) if startY - 10 > 0 else (startX, startY + 20)
    
    # Set the font scale and thickness
    font_scale = 1
    font_thickness = 2

    # Get the text size
    text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

    # Ensure text fits within the box
    while text_size[0] > (endX - startX):
        font_scale -= 0.1
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

    # Put the text on the image
    cv2.putText(image, name, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

    # Generate a unique filename
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]  # Shortened UUID
    output_filename = f"{gender_par}_{timestamp}_{unique_id}.jpg"
    output_image_path = f"/mnt/data/{output_filename}"

    cv2.imwrite(output_image_path, image)
    return output_image_path, box

def gender_and_handle_names(words, image_path, boxes, index=0):
    if index >= len(words):
        return []

    first_name = words[index]
    d = gender.Detector()
    gender_guess = d.get_gender(first_name)
    box_to_image_map = {}

    if gender_guess in ['male', 'mostly_male']:
        output_image_path, box = add_name_to_image(random.choice(male_names), image_path, boxes[index], "male")
    elif gender_guess in ['female', 'mostly_female']:
        output_image_path, box = add_name_to_image(random.choice(female_names), image_path, boxes[index], "female")
    else:  # 'unknown' or 'andy'
        output_image_path, box = add_name_to_image(random.choice(female_names + male_names), image_path, boxes[index], "neutral")

    box_to_image_map[box] = output_image_path
    return box_to_image_map


