import cv2
import numpy as np 

face_classifier = cv2.CascadeClassifier("C:/opencv/haarcascade/haarcascade_frontalface_default.xml")

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    cropped_faces = []
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        cropped_faces.append(cropped_face)
        
    return cropped_faces

cap = cv2.VideoCapture(0)

count = 0

try:
    while True:
        ret, frame = cap.read()
        faces = face_extractor(frame)
        if faces:
            for face in faces:
                count += 1
                face = cv2.resize(face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                
                file_name_path = 'C:/Users/Risha/OneDrive/Desktop/RISHABH all files/opencv project/dataset/' + str(count) + '.jpg'
                cv2.imwrite(file_name_path, face)
                
                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', face)
        else:
            print("Face not found")
        
        if cv2.waitKey(1) == 13 or count == 100:
            break

except KeyboardInterrupt:
    print("Interrupted by user")

cap.release()
cv2.destroyAllWindows()

print("Data set collection completed")
