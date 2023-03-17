import cv2, numpy as np, face_recognition

# load images and covert them to RGB
mopheshi = face_recognition.load_image_file("ImagesBasic/mophe.jpg")
mopheshi = cv2.cvtColor(mopheshi, cv2.COLOR_BGR2RGB)
test = face_recognition.load_image_file("ImagesBasic/mopheTest.jpg")
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

# detect faces in loaded images with face_recognition, [0] returns the first face detected in the image
faceLocations = face_recognition.face_locations(mopheshi)[0]
# encode detected face in 128-dimensions of the provided image parameter
encodeMopheshi = face_recognition.face_encodings(mopheshi)[0]
# draw a rectangle surrounding the detected face in the image provided...
# print(faceLocations) prints (top, right, bottom, left) coordinates of the detected face in the image...
# 2 (last parameter) is the thickness of the rectangle to be drawn around the detected face in the image
cv2.rectangle(mopheshi, (faceLocations[3], faceLocations[0]), (faceLocations[1], faceLocations[2]), (0, 255, 0), 2)

# detect face in test image
faceLocationsTraining = face_recognition.face_locations(test)[0]
encodeTest = face_recognition.face_encodings(test)[0]
cv2.rectangle(test, (faceLocationsTraining[3], faceLocationsTraining[0]),
              (faceLocationsTraining[1], faceLocationsTraining[2]),
              (0, 255, 0), 2)

# compare the encodings to check for similarities/a match recognize faces from trained models/faces...
# 'compare_faces' returns a list of True/False values indicating which known_face_encodings match the
# face encoding to check
result = face_recognition.compare_faces([encodeMopheshi], encodeTest)
# finding best match between the face encodings using distance (lower distance signifies better match)
faceDistance = face_recognition.face_distance([encodeMopheshi], encodeTest)
# write output on a detected face to show result and face distance
cv2.putText(test, f"Result: {result}, Distance: {round(faceDistance[0], 2)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
            (0, 255, 0), 2)
print(f"Result: {result}, Distance: {faceDistance}")

# display loaded images
cv2.imshow("Mopheshi", mopheshi)
cv2.imshow("Mopheshi Test", test)
cv2.waitKey(0)
