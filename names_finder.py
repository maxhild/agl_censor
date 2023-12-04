

def create_combined_phrases_old(ocr_text, box):
    words = ocr_text.split()  # Split the text into words
    combined_phrases = []
    for i in range(len(words)):
        phrase = ' '.join(words[max(0, i-2):i+1])
        combined_phrases.append((phrase, box))
    return combined_phrases

def create_combined_phrases(ocr_texts_with_boxes):
    combined_phrases = []
    combined_box = None
    phrase = ""

    for item in ocr_texts_with_boxes:
        ocr_texts, box = item
        text = ocr_texts[0]['generated_text']  # Assuming there's always at least one dictionary in the list

        # Ensure box is a tuple or list of four elements
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            raise ValueError("Box must be a tuple or list of four elements")

        if not phrase:
            # Start a new phrase
            phrase = text
            combined_box = box
        else:
            # Add to existing phrase and update the box
            phrase += " " + text

            # Ensure combined_box is valid before unpacking
            if not isinstance(combined_box, (tuple, list)) or len(combined_box) != 4:
                raise ValueError("Combined box is not in the correct format")

            startX, startY, endX, endY = combined_box
            new_startX, new_startY, new_endX, new_endY = box
            combined_box = (min(startX, new_startX), min(startY, new_startY),
                            max(endX, new_endX), max(endY, new_endY))

    # Append the final phrase and its box after the loop
    if phrase:
        combined_phrases.append((phrase, combined_box))

    return combined_phrases
