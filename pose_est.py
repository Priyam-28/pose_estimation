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
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 
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

        # this gives the landmarks array further used to get the x,y,z coordinates and angles
        try:
            landmarks = results.pose_landmarks.landmark
            print(landmarks)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Visualize angle
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                            )
        # in the above line, we are drawing the detected lines on the image the results.pose_landmarks gives x,y,z coordinates and the pose connection is used to connecting like nose to inner right and left eye and so on
        cv2.imshow('Mediapipe Feed',image)
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break


for lndmrk in mp_pose.PoseLandmark:
    print(lndmrk)


# logic for calculating the angles