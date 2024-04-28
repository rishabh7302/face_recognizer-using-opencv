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
