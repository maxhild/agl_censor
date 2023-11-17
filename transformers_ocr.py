from transformers import pipeline
from transformers import ViTImageProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance, ImageFilter
from flair_NER import NER_German
from names_gen import gender_and_handle_names
import fitz
import os
import cv2
import numpy as np
import spacy
import re

processor = ViTImageProcessor.from_pretrained('microsoft/trocr-large-str')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-str')
pipe = pipeline("image-to-text", model="microsoft/trocr-large-str")



def process_images_with_OCR_and_NER(file_path, boxes):
    results = []
    print("Processing file:", file_path)

    with open(file_path, 'rb') as file:
        filename = file_path
        file_extension = filename.split('.')[-1].lower()

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

        names = ''  # Initialize 'names' here
        extracted_text = ''

        if file_type == 'pdf':
            # Process each page of the PDF
            for img in images:
                # ... existing image processing code ...
                text = pipe(img)  # OCR
                extracted_text += text
                # Process text and perform NER
                extracted_text = process_text(extracted_text)
                names = NER_German(extracted_text)
                # Call name replacement function
                modified_img = replace_names_in_image(img, names, boxes)  # Function to be implemented
                # Save or process modified_img as needed
        else:
            image = Image.open(file_path)
            # ... existing image processing code ...
            try:
                extracted_text = pipe(image)
            except Exception as e:
                print(f"An error occurred during OCR processing: {e}")
                extracted_text = ""  
            extracted_text = process_text(extracted_text)
            extracted_text = [item['generated_text'] for item in extracted_text if 'generated_text' in item]
            for text in extracted_text:
                text = process_text(text)
                names.= NER_German(str(text))
            names = NER_German(extracted_text)
            # Call name replacement function
            modified_img = gender_and_handle_names(image, names, boxes)  # Function to be implemented
            # Save or process modified_img as needed
      

        result = {
            'filename': filename,
            'file_type': file_type,
            'extracted_text': extracted_text,
            'name_detected': names
        }
        results.append(result)
        print("text detected:", result)

    return results


def download_model(model_name):
    spacy.cli.download(model_name)

#os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata'
#pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
#tessdata_dir_config = '--tessdata-dir "/opt/homebrew/share/tessdata"'


# load a spaCy model
#try:
#    nlp = spacy.load("de_core_news_lg")
#except OSError:
#    print("Model not found. Downloading...")
#    download_model("de_core_news_lg")
#    nlp = spacy.load("de_core_news_lg")


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
    images = []

    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombuffer("RGB", [pix.width, pix.height], pix.samples, "raw", "RGB", 0, 1)

        images.append(img)
        
    return images

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
