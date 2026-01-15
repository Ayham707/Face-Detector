
"""Face detection and preprocessing module using SSD and OpenCV."""

import cv2
from pathlib import Path

# --- Load SSD face detector ---
net = cv2.dnn.readNetFromCaffe(
    "C:\\opencv\\Models\\deploy.prototxt.txt",
   "C:\\opencv\\Models\\res10_300x300_ssd_iter_140000.caffemodel"
)
# Root training folder

folder = Path(r"C:\Users\ayham\Desktop\Downloads\dataset HMI (1)\dataset HMI\training")
# Loop over each person folder
for person_dir in folder.iterdir():
    if not person_dir.is_dir():
        continue

    print(f"\nProcessing person: {person_dir.name}")

    # Loop over images inside person folder
    for img_path in sorted(person_dir.iterdir()):
        if not img_path.is_file():
            continue

        if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Cannot load: {img_path.name}")
            continue

        # --- Create blob ---
        blob = cv2.dnn.blobFromImage(
            img, 1.0, (300, 300),
            (104, 177, 123), False, False
        )

        net.setInput(blob)
        detections = net.forward()

        h, w = img.shape[:2]
        face_box = None

        # --- Find first confident face ---
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.9:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)

                face_box = (x1, y1, x2 - x1, y2 - y1)
                break

        if face_box is None:
            print(f"  No face found: {img_path.name}")
            continue

        x, y, fw, fh = face_box
        x = max(0, x)
        y = max(0, y)
        fw = min(fw, w - x)
        fh = min(fh, h - y)

        face = img[y:y + fh, x:x + fw]

        # --- Preprocessing ---
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (92, 112))
        equalized = cv2.equalizeHist(resized)

        # --- Overwrite original image ---
        cv2.imwrite(str(img_path), equalized)

        print(f"  Processed: {img_path.name}")
