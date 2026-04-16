import cv2
import mediapipe as mp
import os

# ===== CHANGE THIS LETTER EACH TIME =====
LABEL = "A"   # A → B → C → ...

SAVE_DIR = f"ISL/{LABEL}"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opened")
    exit()

count = len(os.listdir(SAVE_DIR))

print(f"Capturing images for alphabet: {LABEL}")
print("Press S to save ONE image")
print("Press Q to quit")
print("Press ESC for emergency exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Draw skeleton if hand detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # UI text
    cv2.putText(frame, f"Alphabet: {LABEL}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.putText(frame, f"Images saved: {count}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, "S=Save | Q=Quit | ESC=Exit", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("ISL Image Capture (Skeleton ON)", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save ONE image on S
    if key == ord('s'):
        img_path = f"{SAVE_DIR}/{count}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        count += 1

    # Quit after one alphabet
    if key == ord('q'):
        print("Finished capturing this alphabet")
        break

    # Emergency exit
    if key == 27:
        print("Emergency exit")
        break

cap.release()
cv2.destroyAllWindows()
