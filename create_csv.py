import cv2
import mediapipe as mp
import os
import csv

# =========================
# CONFIG
# =========================
DATASET_DIR = "ISL"            # Folder with A–Z subfolders
CSV_FILE = "isl_landmarks.csv"

# =========================
# MediaPipe Hands
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.6
)

# =========================
# CSV HEADER
# =========================
header = []
for hand in range(2):  # max 2 hands
    for i in range(21):
        header += [
            f"h{hand}_lm{i}_x",
            f"h{hand}_lm{i}_y",
            f"h{hand}_lm{i}_z"
        ]
header.append("label")

# =========================
# CREATE CSV
# =========================
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for label in sorted(os.listdir(DATASET_DIR)):
        label_path = os.path.join(DATASET_DIR, label)

        if not os.path.isdir(label_path):
            continue

        print(f"Processing letter: {label}")

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                continue

            row = []

            for hand_index in range(2):
                if hand_index < len(results.multi_hand_landmarks):
                    hand_landmarks = results.multi_hand_landmarks[hand_index]
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                else:
                    row.extend([0.0] * 63)  # pad if second hand missing

            row.append(label)
            writer.writerow(row)

print("✅ CSV file created successfully:", CSV_FILE)
