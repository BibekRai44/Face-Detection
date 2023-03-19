import cv2

# Load the pre-trained face detection classifier (Haar cascade)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the image to be detected
img = cv2.imread('img1.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    total_pixels = w * h
    face_pixels = sum(sum(gray[y:y+h, x:x+w] != 0))
    face_percentage = face_pixels / total_pixels * 100
    cv2.putText(img, f"{face_percentage:.2f}%", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

# Display the resulting image
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
