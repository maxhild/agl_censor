from transformers import pipeline
from transformers import ViTImageProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance, ImageFilter
from flair_NER import NER_German
from names_gen import gender_and_handle_names
import fitz
import tempfile
from east_as_a_function import east_text_detection
#from tesseract_as_a_function import tesseract_text_detection
from names_finder import create_combined_phrases
import os
import cv2
import numpy as np
import spacy
import tempfile
import re
import streamlit as st


processor = ViTImageProcessor.from_pretrained('microsoft/trocr-large-str')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-str')
pipe = pipeline("image-to-text", model="microsoft/trocr-large-str")

# new function to perform OCR only on the Boxes provided by EAST

def ocr_on_boxes(image_path, boxes):
    image = Image.open(image_path)
    extracted_text_with_boxes = []
    
    for box in boxes:

        (startX, startY, endX, endY) = box
        cropped_image = image.crop((startX, startY, endX, endY))
        text = pipe(cropped_image)
        print(text)  # See what this output looks like
        #text = pipe(cropped_image)['generated_text']  # Using TROCR on the cropped image
        extracted_text_with_boxes.append((text, box))
        print(f"Box: ", box)


    return extracted_text_with_boxes


def convert_pdf_to_images(pdf_data):
    image_paths = []

    try:
        # Open the PDF file from the byte stream
        with fitz.open(stream=pdf_data, filetype="pdf") as doc:
            for i, page in enumerate(doc):
                pix = page.get_pixmap()
                img = Image.frombuffer("RGB", [pix.width, pix.height], pix.samples, "raw", "RGB", 0, 1)

                # Create a temporary file for the image
                temp_file_path = None
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    img.save(temp_file, format="PNG")
                    temp_file_path = temp_file.name

                if temp_file_path:
                    image_paths.append(temp_file_path)
                    print(f"Converted page {i+1} to image: {temp_file_path}")
                else:
                    print(f"Failed to convert page {i+1} to image.")

    except Exception as e:
        print(f"An error occurred during PDF conversion: {e}")

    return image_paths


def process_images_with_OCR_and_NER(file_path, east_path='frozen_east_text_detection.pb', min_confidence=0.5, width=320, height=320):
    print("Processing file:", file_path)

    # Initialize variables
    modified_images_map = {}
    names_detected = []
    extracted_text = ''

    # Determine the file type
    file_extension = file_path.split('.')[-1].lower()
    mime_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'bmp': 'image/bmp',
        'tiff': 'image/tiff',
        'pdf': 'application/pdf',
    }
    file_type = mime_types.get(file_extension, 'application/octet-stream').split('/')[-1]

    if file_type not in ['jpg', 'jpeg', 'png', 'pdf', 'tiff']:
        raise ValueError('Invalid file type.')

    if file_type == 'pdf':
        image_paths = []

        # Open the PDF file from the file path
        with open(file_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()

            # Convert the entire PDF to images
            image_paths = convert_pdf_to_images(pdf_data)

            # Open the PDF again for text extraction
            with fitz.open(stream=pdf_data, filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                print(text)  # Use print or st.write(text) depending on your output preference

    
        """ 
        with open(file_path, 'rb') as doc:
            text = ""
            for page in doc:
                image_paths = convert_pdf_to_images(page)
                text += page.getText()
            print(text)
        """  

        for img_path in image_paths:
            print("pdf split into images")
            boxes = east_text_detection(img_path, east_path, min_confidence, width, height)
            print("Text boxes detected")

            extracted_text_with_boxes = ocr_on_boxes(img_path, boxes)
            combined_phrases = create_combined_phrases(extracted_text_with_boxes)

            for phrase, phrase_box in combined_phrases:
                processed_text = process_text(phrase)
                entities = NER_German(processed_text)
                for entity in entities:
                    if entity.tag == 'PER':
                        modified_img_path = gender_and_handle_names(processed_text, file_path, phrase_box)
                        modified_images_map[phrase_box] = modified_img_path
                        names_detected.append(entity.text)


    else:
            boxes = east_text_detection(file_path, east_path, min_confidence, width, height)
            print("Text boxes detected")

            extracted_text_with_boxes = ocr_on_boxes(file_path, boxes)
            combined_phrases = create_combined_phrases(extracted_text_with_boxes)

        
            for phrase, phrase_box in combined_phrases:
                processed_text = process_text(phrase)
                entities = NER_German(processed_text)
                if entities:
                    for entity in entities:
                        if entity.tag == 'PER':
                            modified_img_path = gender_and_handle_names(processed_text, file_path, phrase_box)
                            modified_images_map[phrase_box] = modified_img_path
                            names_detected.append(entity.text)

    result = {
        'filename': file_path,
        'file_type': file_type,
        'extracted_text': extracted_text,
        'names_detected': names_detected,
        #'Boxes': boxes
    }

    print("Text detected:", result)
    return result, modified_images_map, boxes


def run_script():
    return "Processing completed"

def replace_unwanted_characters(word):
    # Erlaube nur Zeichen a-z, A-Z, ä, ü, ö, Ä, Ü, Ö, ß, 1-9, . , ' / & % ! " < > + * # ( ) € und -
    allowed_chars = r"[^a-zA-ZäüöÄÜÖß0-9.,'`/&%!\"<>+*#()\€_:-]"
    return re.sub(allowed_chars, '', word)


def clean_newlines(text):
    # Replace two or more consecutive "\n" characters with a single "\n"
    cleaned_text = re.sub(r'\n{2,}', '\n', text)

    # Replace remaining "\n" characters with a space
    cleaned_text = cleaned_text.replace("\n", " ")

    return cleaned_text

def process_image(image, use_mock=False):
    #print("Image size:", image.size)
    #print("Image format:", image.format)
    
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image passed to process_image")

    if use_mock:
        # Return a simple mock image     
        image = Image.new('RGB', (100, 100), color='red')
        return image
    
    # Convert image to OpenCV format
    image = np.array(image)

    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarization
    #_,  image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Scale image
    desired_width = 5000
    aspect_ratio = image.shape[1] / image.shape[0] # width / height
    desired_height = int(desired_width / aspect_ratio)

    image = cv2.resize(image, (desired_width, desired_height), interpolation=cv2.INTER_AREA)    
    
    # Increase contrast using histogram equalization
    #image = cv2.equalizeHist(image)

    # Noise reduction with a Median filter
    #image = cv2.medianBlur(image, 1)

    # Skew correction
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Convert back to PIL.Image format
    image = Image.fromarray(image)
      # Increase contrast
    #enhancer = ImageEnhance.Contrast(image)
    #image = enhancer.enhance(0.9)
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(0.2) 
    
    # Apply blurring filter
    #image = image.filter(ImageFilter.GaussianBlur(radius=0.03))
    
    return image

def convert_pdf_to_images(pdf_data):
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    image_paths = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        img = Image.frombuffer("RGB", [pix.width, pix.height], pix.samples, "raw", "RGB", 0, 1)

        # Save the image to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(temp_file, format="PNG")
        image_paths.append(temp_file.name)

    return image_paths
def scale_coordinates(coords, image_size):
    # Convert the fractional coordinates into actual pixel values
    left = coords['left'] * image_size[0]
    top = coords['top'] * image_size[1]
    right = coords['right'] * image_size[0]
    bottom = coords['bottom'] * image_size[1]
    #print("scaled coordinates:", left, top, right, bottom)
    
    return (left, top, right, bottom)

def crop_image(image, coordinates):
    coordinates = scale_coordinates(coordinates, image.size)
    return image.crop(coordinates)

def extract_coordinates(htmlwithcoords):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(htmlwithcoords, 'html.parser')
    coordinates = []
    for word in soup.find_all(class_='ocrx_word'):
        bbox = word['title'].split(';')[0].split(' ')[1:]
        left, top, right, bottom = map(int, bbox)
        coordinates.append({'left': left, 'top': top, 'right': right, 'bottom': bottom})
    return coordinates

def process_text(extracted_text):
    #extracted_text = clean_newlines(extracted_text)
    return extracted_text
