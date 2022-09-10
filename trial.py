import mediapipe as mp
import pynput
from pynput.keyboard import Key, Controller
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

keyboard = Controller()

# Initiate pose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make Detections
        results = pose.process(image)

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        image = cv2.flip(image, 1)

        # Extracting the coordinates
        lhand = results.pose_landmarks.landmark[15]
        rhand = results.pose_landmarks.landmark[16]
        lshoulder = results.pose_landmarks.landmark[11]
        rshoulder = results.pose_landmarks.landmark[12]

        # Checking if left or right
        if ((lhand.x > lshoulder.x) & (rhand.x > rshoulder.x)):
            print("Left")
            keyboard.release('w')
            keyboard.release('d')
            keyboard.press('a')

        elif ((lhand.x < lshoulder.x) & (rhand.x < rshoulder.x)):
            print("Right")
            keyboard.release('a')
            keyboard.release('w')
            keyboard.press('d')

        elif ((lhand.y < lshoulder.y) & (rhand.y < rshoulder.y)):
            print("Up")
            keyboard.release('a')
            keyboard.release('d')
            keyboard.press('w')

        elif ((lhand.y < lshoulder.y) & (rhand.y < rshoulder.y)):
            keyboard.release('a')
            keyboard.release('d')
            keyboard.release('w')

        if ((lhand.y < lshoulder.y) & (rhand.y < rshoulder.y) & (lhand.x < rhand.x)):
            print("Lol")
            break
        # Draw connections

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
