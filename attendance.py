import os
from datetime import datetime

import cv2
import face_recognition
import numpy as np

path = "ImagesAttendance"
images = []
classNames = []
# load images in 'ImagesAttendance' to a list
imagesList = os.listdir(path)
print(imagesList)

# load images from imagesList
for klass in imagesList:
    currentImage = cv2.imread(f"{path}/{klass}")  # load images from path and add them to list of images
    images.append(currentImage)
    classNames.append(os.path.splitext(klass)[0])  # return file name without its extension


# print(classNames)


# function to convert loaded images from 'path' to RGB and encode detected faces in 128-dimensions of the provided
# image parameter
def findEncodings(images):
    encodeList = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert image to RGB as usual
        encode = face_recognition.face_encodings(image)[0]  # encode the {first detected face ([0])} image
        encodeList.append(encode)

    return encodeList


def markAttendance(name):
    with open("Files/attendance.csv", "r+") as file:
        studentList = file.readlines()
        nameList = []
        for line in studentList:
            entry = line.split(',')
            nameList.append(entry[0])  # add names of students to nameList, entry[0] = Name (first element)

        if name not in nameList:
            now = datetime.now()  # current device time
            time = now.strftime("%H:%M:%S")  # format device time to hour, minute and seconds
            file.writelines(f"\n{name},{time}")  # append new entry to next line under heading of attendance.csv file


encodeListForKnownFaces = findEncodings(images)
# print(f"Encoding complete with {len(encodeListForKnownFaces)} images encoded...")

# enable webcam to get input images to check for a match and face recognition...
capture = cv2.VideoCapture(0)

while True:
    success, image = capture.read()
    # resize image gotten from video to speed up the process... (0.25 = .25 = 1/4)
    smallImage = cv2.resize(image, (0, 0,), None, .25, .25)
    smallImage = cv2.cvtColor(smallImage, cv2.COLOR_BGR2RGB)

    facesInCurrentFrame = face_recognition.face_locations(smallImage)  # locations of all detected faces
    encodingsOfCurrentFrame = face_recognition.face_encodings(smallImage,
                                                              facesInCurrentFrame)  # encodings of all detected faces

    # iterate through all face locations and image encodings to find a match for facial recognition
    # face locations and image encodings are zipped to be checked in the same iteration
    for encodeFace, faceLocation in zip(encodingsOfCurrentFrame, facesInCurrentFrame):
        # compare a list of face encodings against a candidate encoding to see if they match
        match = face_recognition.compare_faces(encodeListForKnownFaces, encodeFace)
        # given a list of face encodings, compare them to a known face encoding and get a euclidean distance for each
        # comparison face. The distance tells you how similar the faces are, the smaller the distance the more
        # similar the faces are
        distance = face_recognition.face_distance(encodeListForKnownFaces, encodeFace)
        # print(distance)
        # find the indices of the minimum values along an axis, in case of multiple occurrences of the minimum
        # values, the indices corresponding to the first occurrence are returned
        matchIndex = np.argmin(distance)

        if match[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # multiply all angles by 4 to regenerate original image
            # from small image (0.25 or .25 or 1/4) to normal scale
            cv2.rectangle(image, (x1, y1),
                          (x2, y1), (0, 255, 0),
                          2)  # draw a green rectangle surrounding the detected face(s) in WebCam...
            cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0),
                          cv2.FILLED)  # green rectangle that will contain the name if the detected face at the
            # bottom of the bounding rectangle
            cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow("WebCam", image)
    cv2.waitKey(1)
