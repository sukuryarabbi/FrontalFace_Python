import cv2
import numpy as np

capture = cv2.VideoCapture(0)

while True:

    ret,frame = capture.read()


    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier("HaarCascade/frontalface.xml")

    faces = face_cascade.detectMultiScale(gray , 1.3 , 4)

    for x,y,w,h in faces : 

        cv2.rectangle(frame , (x,y) , (x+w,y+h) , (0,255,0) , 3)

    cv2.imshow("ekran" , frame)

    kInp = cv2.waitKey(1)

    if(kInp == ord("q") ) : 

        break

capture.release()
cv2.destroyAllWindows()

    



