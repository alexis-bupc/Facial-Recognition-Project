import cv2 as cv 
import numpy as np
  
# Load the cascade  
haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Aaron', 'Alexis', 'Caren', 'Florenz', 'Karen', 'Nichole']
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

# To capture video from webcam.   
cap = cv.VideoCapture(0)  
text = "Face Detected"

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

while True:  
    # Read the frame  
    _, img = cap.read()  
  
    # Convert to grayscale  
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
  
    # Detect the faces  
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
  
    # Draw the rectangle around each face  
    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+h]
        label, confidence = face_recognizer.predict(faces_roi)  
        print(f'Label = {people[label]} with a confidence of {confidence}')
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 555, 0), 2)
        cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    
    # Display  
    cv.imshow('Video', img)  
  
    # Stop if escape key is pressed  
    k = cv.waitKey(30) & 0xff  
    if k==27:  
        break  
          
# Release the VideoCapture object  
cap.release()  