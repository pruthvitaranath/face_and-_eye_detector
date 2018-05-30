import cv2
import numpy as np

#Face and eye classifier using the haar cascade classifier

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

#Function to detect face and eye

def detector(img):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_img,1.3,5)

    if faces is ():
        print ("")

    for (x,y,w,h) in faces:
        x = x-10
        w = w+10
        y = y-10
        h = h+10
        cv2.rectangle(img,(x,y),(x+w,y+h),(12,127,255),2)

        #Cropping the face to detect Eyes
        
        new_image = img[y:y+h,x:x+w]
        new_gray_img = gray_img[y:y+h,x:x+w]
        eyes = eye_classifier.detectMultiScale(new_gray_img)

        for i,j,k,l in eyes:
            cv2.rectangle(new_image,(i,j),(i+k,j+l),(255,127,127),2)
        
    return img

#Initializing the web cam

cam = cv2.VideoCapture(0)

while True:
    ret,frame = cam.read()
    cv2.imshow('Detected Face and Eye', detector(frame))
    cv2.waitKey(0)


cam.release()
cv2.destroyAllWindows()
