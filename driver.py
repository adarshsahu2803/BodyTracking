import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#cap = cv2.VideoCapture(0)


# Initiate pose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # while cap.isOpened():
    #     ret, frame = cap.read()
    for i in range(0, 5):

        frame = cv2.imread("C:/Users/adars/OneDrive/Desktop/testtttt/a" + str(i) + ".png")
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make Detections
        results = pose.process(image)
        bg = cv2.imread("C:/Users/adars/OneDrive/Desktop/testtttt/b.jpg")

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        image = cv2.flip(image, 1)
        # bg = cv2.flip(bg, 1)

        # Extracting the coordinates
        lhand = results.pose_landmarks.landmark[15]
        rhand = results.pose_landmarks.landmark[16]
        lshoulder = results.pose_landmarks.landmark[11]
        rshoulder = results.pose_landmarks.landmark[12]

        # print(lhand)

        # Checking if left or right
        if ((lhand.x > lshoulder.x) & (rhand.x > rshoulder.x)):
            print("Left")

        elif ((lhand.x < lshoulder.x) & (rhand.x < rshoulder.x)):
            print("Right")

        elif ((lhand.y < lshoulder.y) & (rhand.y < rshoulder.y)):
            print("Up")

        if ((lhand.y > lshoulder.y) & (rhand.y < rshoulder.y) & (lhand.x < rhand.x)):
            print("Lol")
            break

        # Draw connection
        cv2.imshow('Raw Webcam Feed', image)
        cv2.waitKey(1000)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# cap.release()
cv2.destroyAllWindows()
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# cap = cv2.VideoCapture(0)

# Initiate pose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    # while cap.isOpened():
    #     ret, frame = cap.read()
    for i in range(0, 5):

        frame = cv2.imread("C:/Users/adars/OneDrive/Desktop/testtttt/a" + str(i) + ".png")
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make Detections
        results = pose.process(image)
        bg = cv2.imread("C:/Users/adars/OneDrive/Desktop/testtttt/b.jpg")

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        image = cv2.flip(image, 1)
        # bg = cv2.flip(bg, 1)

        #Extracting the coordinates
        lhand = results.pose_landmarks.landmark[15]
        rhand= results.pose_landmarks.landmark[16]
        lshoulder = results.pose_landmarks.landmark[11]
        rshoulder = results.pose_landmarks.landmark[12]

        # print(lhand)

        #Checking if left or right
        if((lhand.x> lshoulder.x)&(rhand.x> rshoulder.x)):
            print("Left")

        elif((lhand.x< lshoulder.x)&(rhand.x< rshoulder.x)):
            print("Right")

        elif((lhand.y< lshoulder.y)&(rhand.y< rshoulder.y)):
            print("Up")

        if((lhand.y> lshoulder.y)&(rhand.y< rshoulder.y)&(lhand.x< rhand.x)):
            print("Lol")
            break

        # Draw connection
        cv2.imshow('Raw Webcam Feed', image)
        cv2.waitKey(1000)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# cap.release()
cv2.destroyAllWindows()
