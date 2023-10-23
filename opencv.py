import cv2
import numpy as np

imgAll = cv2.imread("Resimler/imgAll.png")
img1 = cv2.imread("Resimler/a1.jpg")
FrontalFace = cv2.imread("Resimler/FrontalFacess.jpg")

face_cascade = cv2.CascadeClassifier("HaarCascade/frontalface.xml")

griFrontalFace = cv2.cvtColor(FrontalFace,cv2.COLOR_BGR2GRAY)
gri1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
griImgAll = cv2.cvtColor(imgAll , cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(griFrontalFace , 1.3 , 4)
kart = face_cascade.detectMultiScale(gri1 , 1.3 , 4)

for x,y,w,h in faces : 

    cv2.rectangle(FrontalFace , (x,y) , (x+w,y+h) ,(0,255,255) , 3 )


cv2.imshow("frontalface" , FrontalFace)

cv2.waitKey(0)

cv2.destroyAllWindows()