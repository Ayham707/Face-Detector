import cv2

# Use OpenCV's Haar cascade 
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if cascade loaded properly
if cascade.empty():
    print("Error: Failed to load cascade classifier!")
    exit()

img = cv2.imread(r"C:\Users\ayham\Face-Detector\test.jpg")

# Check if image loaded properly
if img is None:
    print("Error: Failed to load image!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# More strict parameters to reduce false positives
faces = cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,      # Increased from 1.1 for fewer scales
    minNeighbors=10,      # Increased from 5 for stricter detection
    minSize=(40, 40),     # Increased from (24,24) for larger minimum
    flags=cv2.CASCADE_SCALE_IMAGE
)

print(f"Detected {len(faces)} face(s)")

# Filter out detections with unrealistic aspect ratios
filtered_faces = []
for (x, y, w, h) in faces:
    aspect_ratio = w / float(h)
    # Faces typically have aspect ratio between 0.7 and 1.3
    if 0.7 <= aspect_ratio <= 1.3:
        filtered_faces.append((x, y, w, h))
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    else:
        print(f"Rejected detection at ({x},{y}) with aspect ratio {aspect_ratio:.2f}")

print(f"After filtering: {len(filtered_faces)} valid face(s)")

# Resize image to fit screen better
scale_percent = 20  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("Faces", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
