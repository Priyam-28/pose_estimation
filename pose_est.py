import cv2
import mediapipe as mp
import numpy as np
# for showing the detected lines
mp_drawing = mp.solutions.drawing_utils
# getting the video and estimating the pose
mp_pose = mp.solutions.pose

cap=cv2.VideoCapture(0)
# while cap.isOpened():
#     ret,frame=cap.read()
#     cv2.imshow('Mediapipe Feed',frame)
#     if cv2.waitKey(10) & 0xFF==ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret,frame=cap.read()
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False

        # make detection
        results=pose.process(image)

        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
        # in the above line, we are drawing the detected lines on the image the results.pose_landmarks gives x,y,z coordinates
        cv2.imshow('Mediapipe Feed',image)
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break