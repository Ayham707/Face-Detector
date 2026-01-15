import cv2
from pathlib import Path

# --- Load SSD face detector ---
net = cv2.dnn.readNetFromCaffe(
    "C:\\opencv\\Models\\deploy.prototxt.txt",
    "C:\\opencv\\Models\\res10_300x300_ssd_iter_140000.caffemodel"
)

# --- Positive images ---
pos_folder = "C:\\Users\\ayham\\Face-Detector\\dataset\\positive"
outfile = open("positives.txt", "w")
files = sorted([f for f in Path(pos_folder).iterdir() if f.is_file()])

for entry in files:
    path = str(entry)
    filename = entry.name

    # Only process image files
    lower = filename.lower()
    if not (lower.endswith(".jpg") or lower.endswith(".png") or lower.endswith(".jpeg")):
        continue

    img = cv2.imread(path)
    if img is None:
        print(f"Could not load image: {filename}")
        continue

    # --- Resize image for display ---
    h, w = img.shape[:2]
    scale = min(1000.0 / w, 800.0 / h)
    displayImg = cv2.resize(img, None, fx=scale, fy=scale)

    # --- Create blob and detect faces ---
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 177, 123), False, False)
    net.setInput(blob)
    detections = net.forward()

    faces = []
    numDetections = detections.shape[2]

    for i in range(numDetections):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.90:  # High confidence threshold
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            faces.append((x1, y1, x2 - x1, y2 - y1))
            cv2.rectangle(displayImg, (int(x1 * scale), int(y1 * scale)),
                          (int(x2 * scale), int(y2 * scale)), (0, 255, 0), 2)

    # --- Write to positives.txt ---
    outfile.write(f"{path} {len(faces)}")
    for x, y, w_face, h_face in faces:
        outfile.write(f" {x} {y} {w_face} {h_face}")
    outfile.write("\n")

    # --- Show result ---
    cv2.imshow("SSD Face Detection", displayImg)
    print(f"Image: {filename}  Faces: {len(faces)}")
    print("Press any key for next, or 'q' to stop.")

    key = cv2.waitKey(0)

    # --- Safely destroy window ---
    try:
        cv2.destroyWindow("SSD Face Detection")
    except cv2.error:
        pass

    if key == ord('q') or key == ord('Q'):
        break

outfile.close()
print("positives.txt successfully generated!")

# --- Negative images ---
neg_folder = "C:\\Users\\ayham\\Face-Detector\\dataset\\negative"
out = open("negatives.txt", "w")
files = sorted([f for f in Path(neg_folder).iterdir() if f.is_file()])

for entry in files:
    path = str(entry)
    filename = entry.name
    lower = filename.lower()
    if lower.endswith(".jpg") or lower.endswith(".png") or lower.endswith(".jpeg"):
        out.write(f"{path}\n")

out.close()
print("negatives.txt successfully created!")
