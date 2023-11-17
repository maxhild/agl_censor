import cv2
import numpy as np
import random
import gender_guesser.detector as gender


def gender_and_handle_names(words, image_path, boxes, index=0):
    if index >= len(words):
        return

    first_name = words[index]
    d = gender.Detector()
    gender_guess = d.get_gender(first_name)

    if gender_guess == 'unknown':
        if index == len(words) - 1:
            return add_neutral_name_to_image(first_name, image_path, boxes)
        else:
            return gender_and_handle_names(words, index + 1, image_path, boxes)

    elif gender_guess == 'andy':
        add_neutral_name_to_image(first_name, image_path, boxes)
    elif gender_guess in ['male', 'mostly_male']:
        add_male_name_to_image(first_name, image_path, boxes)
    elif gender_guess in ['female', 'mostly_female']:
        add_female_name_to_image(first_name, image_path, boxes)
    else:
        print("Unrecognized gender category.")


# Load the names from the file
with open('names_dict/first_and_last_name_female.txt', 'r') as file:
    female_names = file.readlines()

# Clean names to remove any newlines
female_names = [name.strip() for name in female_names]

# Load the names from the file
with open('/names_dict/first_and_last_name_male.txt', 'r') as file:
    male_names = file.readlines()

# Clean names to remove any newlines
male_names = [name.strip() for name in male_names]

# Function to add text to an image in a given bounding box
def add_female_name_to_image(image_path, boxes, names_list=female_names):
    # Load the image
    image = cv2.imread(image_path)

    # Choose a random name from the list
    name = random.choice(names_list)

    # Choose the first box to put the text in (for demonstration purposes)
    # If there are no boxes or text is not detected, we won't add any text
    if boxes:
        (startX, startY, endX, endY) = boxes[0]
        
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

    # Save the modified image
    output_image_path = '/mnt/data/modified_lebron_james.jpg'
    cv2.imwrite(output_image_path, image)
    gender_par = "female"
    
    return output_image_path, gender_par

# Function to add text to an image in a given bounding box
def add_male_name_to_image(image_path, boxes, names_list=male_names):
    # Load the image
    image = cv2.imread(image_path)

    # Choose a random name from the list
    name = random.choice(names_list)

    # Choose the first box to put the text in (for demonstration purposes)
    # If there are no boxes or text is not detected, we won't add any text
    if boxes:
        (startX, startY, endX, endY) = boxes[0]
        
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

    # Save the modified image
    output_image_path = '/mnt/data/modified_lebron_james.jpg'
    cv2.imwrite(output_image_path, image)
    gender_par = "male"

    
    return output_image_path, gender_par

def add_neutral_name_to_image(image_path, boxes, names_list=female_names):
# Function to add text to an image in a given bounding box
    # Load the image
    image = cv2.imread(image_path)

    # Choose a random name from the list
    name = random.choice(names_list)

    # Choose the first box to put the text in (for demonstration purposes)
    # If there are no boxes or text is not detected, we won't add any text
    if boxes:
        (startX, startY, endX, endY) = boxes[0]
        
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

    # Save the modified image
    output_image_path = '/mnt/data/modified_lebron_james.jpg'
    cv2.imwrite(output_image_path, image)
    gender_par = "neutral"
    
    return output_image_path, gender_par