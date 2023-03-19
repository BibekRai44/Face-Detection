import cv2
import numpy as np
from tensorflow.keras.models import load_model

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]

img = cv2.imread('img5.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    total_pixels = w * h
    face_pixels = sum(sum(gray[y:y+h, x:x+w] != 0))
    face_percentage = face_pixels / total_pixels * 100
    cv2.putText(img, f"{face_percentage:.2f}%", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (64, 64))
    roi_gray = np.expand_dims(roi_gray, axis=-1)
    roi_gray = np.expand_dims(roi_gray, axis=0)
    roi_gray = roi_gray / 255.0

    preds = emotion_model.predict(roi_gray)[0]
    emotion_label = EMOTIONS[np.argmax(preds)]
    emotion_confidence = preds[np.argmax(preds)]

cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
