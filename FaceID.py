# required libraries

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import glob

# Loading Train dataset

path = " "
images = []
className = []
myList = os.listdir(path)
print(myList)

# Extracting names from a file

for cl in myList:
  curImg = cv2.imread(f"{path}/{cl}")
  images.append(curImg)
  className.append(os.path.splitext(cl)[0])
print(className)