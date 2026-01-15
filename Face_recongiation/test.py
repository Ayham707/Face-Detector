import cv2
import os
import numpy as np

# ---------- Load cascade ----------
cascade_path = r"C:\Users\ayham\Face-Detector\classifier\cascade.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise IOError("Failed to load cascade")

# ---------- Test folder ----------
folder = r"C:\Users\ayham\Face-Detector\dataset\test"

# ---------- CLAHE ----------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ---------- Validation ----------
def is_valid_face(gray_roi):
    if gray_roi.size == 0:
        return False

    # Texture check (faces are not flat)
    if gray_roi.std() < 15:
        return False

    h, w = gray_roi.shape
    aspect = w / float(h)
    if aspect < 0.75 or aspect > 1.3:
        return False

    return True

# ---------- Loop ----------
for filename in os.listdir(folder):
    path = os.path.join(folder, filename)
    if not path.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img = cv2.imread(path)
    if img is None:
        continue

    # Resize for consistency
    scale = 1000.0 / img.shape[1]
    scale = min(scale, 1.0)
    img = cv2.resize(img, None, fx=scale, fy=scale)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)

    all_faces = []

    # ---------- Multi-scale detection ----------
    for factor in [1.0, 1.3]:
        resized = cv2.resize(gray, None, fx=factor, fy=factor)

        faces = face_cascade.detectMultiScale(
            resized,
            scaleFactor=1.08,
            minNeighbors=6,
            minSize=(24, 24),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            all_faces.append((
                int(x / factor),
                int(y / factor),
                int(w / factor),
                int(h / factor)
            ))

    # ---------- Group rectangles ----------
    if len(all_faces) > 0:
        rects, _ = cv2.groupRectangles(all_faces, groupThreshold=2, eps=0.2)
    else:
        rects = []

    # ---------- Validation ----------
    final_faces = []
    for (x, y, w, h) in rects:
        roi = gray[y:y+h, x:x+w]
        if is_valid_face(roi):
            final_faces.append((x, y, w, h))

    print(f"{filename} → {len(final_faces)} faces detected")

    # ---------- Draw ----------
    for (x, y, w, h) in final_faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Result", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
