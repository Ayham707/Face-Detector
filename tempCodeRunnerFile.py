import cv2
import os
import numpy as np

# --- Helper function for ends_with ---
def ends_with(string, suffixes):
    string = string.lower()
    return any(string.endswith(s.lower()) for s in suffixes)

# --- Load your trained Viola-Jones cascade ---
cascade_path = r"C:\Users\ayham\Face-Detector\classifier\cascade.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print(f"Could not load cascade: {cascade_path}")
    exit(-1)

# --- Folder with test images ---
folder = r"C:\Users\ayham\Face-Detector\dataset\test"

# --- Function to filter overlapping detections (Non-Maximum Suppression) ---
def non_max_suppression(boxes, overlap_thresh=0.1):  # REDUCED from 0.3
    """
    Remove overlapping bounding boxes, keeping only the best ones
    """
    if len(boxes) == 0:
        return []
    
    # Convert to float for division
    boxes = np.array(boxes, dtype="float")
    
    # Extract coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    # Compute area of each box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by bottom-right y-coordinate
    idxs = np.argsort(y2)
    
    pick = []
    
    while len(idxs) > 0:
        # Grab the last index and add it to picked list
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # Find overlap with remaining boxes
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # Compute width and height of overlap
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # Compute overlap ratio
        overlap = (w * h) / area[idxs[:last]]
        
        # Delete overlapping boxes
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))
    
    # Return selected boxes as integers
    return boxes[pick].astype("int")

# --- Function to validate face detection (MORE STRICT) ---
def is_valid_face(gray_roi):
    """
    Additional validation to reduce false positives
    Check if detected region has face-like characteristics
    """
    # Check if region has sufficient contrast/variance
    if gray_roi.std() < 20:  # INCREASED from 15 - more strict
        return False
    
    # Check aspect ratio (faces are typically not too elongated)
    h, w = gray_roi.shape
    aspect_ratio = w / float(h)
    if aspect_ratio < 0.7 or aspect_ratio > 1.5:  # MORE STRICT range
        return False
    
    # Check if region has reasonable edge density (faces have edges)
    edges = cv2.Canny(gray_roi, 50, 150)
    edge_density = np.sum(edges > 0) / (w * h)
    if edge_density < 0.05 or edge_density > 0.4:  # Too few or too many edges
        return False
    
    return True

# --- Loop through all files in the folder ---
for filename in os.listdir(folder):
    path = os.path.join(folder, filename)
    
    if not os.path.isfile(path):
        continue

    if not ends_with(filename, (".jpg", ".jpeg", ".png")):
        continue

    # --- Load image ---
    img = cv2.imread(path)
    if img is None:
        print(f"Could not load image: {filename}")
        continue

    # --- Optional: resize image ---
    scale = 800.0 / img.shape[1]
    if scale > 1.0:
        scale = 1.0
    resized = cv2.resize(img, None, fx=scale, fy=scale)

    # --- Convert to grayscale ---
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # --- Apply histogram equalization for better contrast ---
    gray = cv2.equalizeHist(gray)

    # --- Detect faces with VERY STRICT parameters ---
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,        # INCREASED from 1.05 - larger steps, faster but less sensitive
        minNeighbors=1
        ,        # INCREASED from 10 - much stricter (try 12-20)
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(40, 40),       # INCREASED minimum size
        maxSize=(250, 250)      # REDUCED maximum size
    )

    # --- Apply Non-Maximum Suppression with stricter threshold ---
    if len(faces) > 0:
        faces = non_max_suppression(faces, overlap_thresh=0.2)  # More aggressive
    
    # --- Additional validation: filter out unlikely faces ---
    validated_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        if is_valid_face(roi_gray):
            validated_faces.append((x, y, w, h))
    
    faces = validated_faces

    print(f"{filename} → {len(faces)} faces detected")

    # --- Draw bounding boxes ---
    for (x, y, w, h) in faces:
        cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Optional: add confidence indicator
        cv2.putText(resized, "Face", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- Show result ---
    cv2.imshow("Result", resized)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()