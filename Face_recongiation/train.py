import cv2
import numpy as np
from pathlib import Path

root = Path(r"C:\Users\ayham\Desktop\Downloads\dataset HMI (1)\dataset HMI\training")

images = []
labels = []
label_names = []

current_label = 0

for person_dir in root.iterdir():
    if not person_dir.is_dir():
        continue

    person_name = person_dir.name
    label_names.append(person_name)

    print(f"Laad persoon: {person_name} -> label {current_label}")

    for img_path in person_dir.iterdir():
        if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (92, 112))

        images.append(img)
        labels.append(current_label)

    current_label += 1

print(f"Aantal geladen afbeeldingen: {len(images)}")

# 🔴 REQUIRED FIX
labels = np.array(labels, dtype=np.int32)

# Train EigenFaces
model = cv2.face.EigenFaceRecognizer_create()
model.train(images, labels)

model.save(r"C:\Users\ayham\eigenfaces_model.xml")

with open(r"C:\Users\ayham\labelmap.txt", "w", encoding="utf-8") as f:
    for i, name in enumerate(label_names):
        f.write(f"{i};{name}\n")

print("Model en labelmap opgeslagen.")
