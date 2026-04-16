import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,   # 🔴 BOTH HANDS
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

print("Press P to print landmarks")
print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF

    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Print landmarks only when P is pressed
            if key == ord('p'):
                print(f"\nHAND {hand_index + 1}")
                h, w, _ = frame.shape
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(f"ID {id}: x={cx}, y={cy}, z={lm.z:.3f}")
                print("-" * 30)

    cv2.imshow("Hand Landmarks (1 & 2 Hands)", frame)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
