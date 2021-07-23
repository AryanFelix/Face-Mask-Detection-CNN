import numpy as np
import cv2

from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

model = load_model("FaceMaskModel.h5")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') 

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    success, temp = cap.read()
    img = cv2.flip(temp, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:  
        faceFrame = img[y: y + h, x: x + w]
        faceFrame = cv2.cvtColor(faceFrame, cv2.COLOR_BGR2RGB)
        faceFrame = cv2.resize(faceFrame, (128, 128))
        faceFrame = image.img_to_array(faceFrame)
        faceFrame = np.expand_dims(faceFrame, axis=0)
        faceFrame = faceFrame/255.0
        predictions = model.predict(faceFrame)
        print(predictions[0])
        (incorrectMask, withMask, withoutMask) = predictions[0]
        if incorrectMask > withMask:
            if incorrectMask > withoutMask:
                label = "WEAR MASK PROPERLY!"
            else:
                label = "No Mask Detected"
        else:
            if withMask > withoutMask:
                label = "Well Masked"
            else:
                label = "No Mask Detected"
        if label == "Well Masked":
            color = (0, 255, 0)
        elif label=="No Mask Detected":
            color = (0, 0, 255)
        else:
            color = (255, 140, 0)
        label = "{}: {:.2f}%".format(label, max(withMask, withoutMask) * 100)
        cv2.putText(img, label, (x, y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(img, (x, y), (x + w, y + h),color, 2)
    cv2.imshow("Press 'x' to Exit!", img)
    if cv2.waitKey(1) & 0xFF ==ord('x'):
        break
cap.release()
cv2.destroyAllWindows()