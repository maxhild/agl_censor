import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_dominant_color(image, box, k=1):
    # Crop the background area around the text box
    (startX, startY, endX, endY) = box
    margin = 10  # You can adjust the size of the margin
    cropped_img = image[max(startY - margin, 0):min(endY + margin, image.shape[0]),
                        max(startX - margin, 0):min(endX + margin, image.shape[1])]
    
    # Convert to a color space that may improve color clustering
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LAB)

    # Reshape the image to be a list of pixels
    pixels = cropped_img.reshape((cropped_img.shape[0] * cropped_img.shape[1], 3))

    # Use KMeans clustering to find the most prevalent color
    clt = KMeans(n_clusters=k, n_init=10)
    clt.fit(pixels)

    # Find the most prevalent cluster center
    numLabels = np.arange(0, k + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Find the largest cluster
    dominant_color = clt.cluster_centers_[np.argmax(hist)]

    # Convert dominant color back to BGR color space for OpenCV to display
    dominant_color = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_LAB2BGR)[0][0]
    
    return tuple(map(int, dominant_color))

def extend_boxes_if_needed(image, boxes, extension_margin=10, color_threshold=30):
    extended_boxes = []
    for box in boxes:
        (startX, startY, endX, endY) = box

        # Get the dominant color of the current box
        dominant_color = get_dominant_color(image, box)

        # Check the color signal around the box and decide if extension is needed
        # Check above the box
        if startY - extension_margin > 0:
            upper_region_color = get_dominant_color(image, (startX, startY - extension_margin, endX, startY))
            if np.linalg.norm(np.array(upper_region_color) - np.array(dominant_color)) > color_threshold:
                startY = max(startY - extension_margin, 0)

        # Check below the box
        if endY + extension_margin < image.shape[0]:
            lower_region_color = get_dominant_color(image, (startX, endY, endX, endY + extension_margin))
            if np.linalg.norm(np.array(lower_region_color) - np.array(dominant_color)) > color_threshold:
                endY = min(endY + extension_margin, image.shape[0])

        # Check left of the box
        if startX - extension_margin > 0:
            left_region_color = get_dominant_color(image, (startX - extension_margin, startY, startX, endY))
            if np.linalg.norm(np.array(left_region_color) - np.array(dominant_color)) > color_threshold:
                startX = max(startX - extension_margin, 0)

        # Check right of the box
        if endX + extension_margin < image.shape[1]:
            right_region_color = get_dominant_color(image, (endX, startY, endX + extension_margin, endY))
            if np.linalg.norm(np.array(right_region_color) - np.array(dominant_color)) > color_threshold:
                endX = min(endX + extension_margin, image.shape[1])

        # Add the possibly extended box to the list
        extended_boxes.append((startX, startY, endX, endY))

    return extended_boxes

# Example of using the function on the first box

