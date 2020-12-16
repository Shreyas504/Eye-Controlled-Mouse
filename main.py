import cv2 
import numpy as np
import dlib
from math import hypot

shapepredictor = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapepredictor)

video = cv2.VideoCapture(0)

def midpoint(p1,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_SIMPLEX

def get_Blinking_Ratio(eye_points,facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[1]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[5]))

    horz_line = cv2.line(frame,left_point,right_point,(255,0,0),2)
    vert_line = cv2.line(frame,center_top,center_bottom,(255,0,0),2)

    vert_line_length = hypot((center_top[0]-center_bottom[0]), (center_top[1]-center_bottom[1]))
    horz_line_length = hypot((left_point[0]-right_point[0]), (left_point[1]-right_point[1]))

    ratio = horz_line_length / vert_line_length
    return ratio

while(True): 
      

    # Capture the video frame 
    # by frame 
    ret, frame = video.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Face detection
    faces = detector(gray)
    #Face Rectangle
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
        
        #Blink Detection
        landmarks = predictor(gray,face)

        left_eye_ratio = get_Blinking_Ratio([36,37,38,39,40,41],landmarks)
        right_eye_ratio = get_Blinking_Ratio([42,43,44,45,46,47],landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        
        if blinking_ratio > 4:
            cv2.putText(frame, "BLINKING", (50,150), font, 7, (255,255,0))

        #Gaze Detection
        
    cv2.imshow("Frame",frame)
    # Display the resulting frame 
    
       
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# After the loop release the cap object 
video.release()
# Destroy all the windows 
cv2.destroyAllWindows() 
