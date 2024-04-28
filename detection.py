import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = 'C:/Users/Risha/OneDrive/Desktop/RISHABH all files/opencv project/dataset/'

onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [], []

for i, file in enumerate(onlyfiles):
    image_path = join(data_path, onlyfiles[i])
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if images is not None:
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

# Check if training data and labels are not empty
if len(Training_Data) > 0 and len(Labels) > 0:
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print("Dataset model training completed")
else:
    print("No training data or labels available. Please check your data.")

###########################################################################################

face_classifier = cv2.CascadeClassifier("C:/opencv/haarcascade/haarcascade_frontalface_default.xml")

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:  # Check if no face is detected
        return img, []
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)   # Set color and width of the box
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
        
        return img, roi
  
# Open video and capture images  
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)
    
    # Check if a face is detected
    if len(face) != 0:
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)
            
            if result[1] < 500:
                confidence = int(100 * (1 - (result[1] / 300)))
            
            if confidence > 82:
                cv2.putText(image, "Aditya Negi", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, "Rishabh Negi", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Face cropper', image)
                
            else:
                cv2.putText(image, "unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)
                
        except:
            cv2.putText(image, "Face not found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("face cropper", image)
            pass
    
    # Close webcam popup if Enter key is pressed
    if cv2.waitKey(1) == 13:
        break
   
# Release capture
cap.release()
cv2.destroyAllWindows()
