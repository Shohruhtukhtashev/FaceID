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

# csv file create, write and update

with open("information.csv","w") as f:
  f.writelines('Name,Group,Course,Time')

# Training process

def findEncodings(image):
  encodeList = []
  for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    encodeList.append(encode)
  return encodeList

def markAttendance(name):
  info = ["group","course"]
  student = ["Students_name"]
  summary = dict.fromkeys(student,info)
  with open("/content/drive/MyDrive/Colab Notebooks/Face/information.csv","r+") as f:
    nameList = []
    myDataList = f.readline()
    for line in myDataList:
      entry = line.split(',')
      nameList.append(entry[0])
    if name not in nameList:
      now = datetime.now()
      dtString = now.strftime('%H:%M:%S')
      f.writelines(f"\n{name},{','.join(summary[name])},{dtString}")

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = glob.glob("/content/drive/MyDrive/Colab Notebooks/Face/test/*")

for i in range(len(cap)):
  img = cv2.imread(cap[i])
  imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)


  facesCurFrame = face_recognition.face_locations(imgS)
  encodesCutFrame = face_recognition.face_encodings(imgS, facesCurFrame)


  for encodeFace, faceLoc in zip(encodesCutFrame, facesCurFrame):
    matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
    faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
    print(faceDis)
    matchIndex = np.argmin(faceDis)

    if matches[matchIndex]:
      name = className[matchIndex]
      print(name)
      y1,x2,y2,x1 = faceLoc
      y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
      cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
      cv2.rectangle(img, (x2,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
      cv2.putText(imgS,name,(x1,y2),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
      markAttendance(name)
  cv2_imshow(imgS)