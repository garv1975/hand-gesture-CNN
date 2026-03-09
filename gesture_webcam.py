import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("gesture_model.h5")

gestures = ['palm','l','fist','fist_moved','thumb','index','ok','palm_moved','c','down']

cap = cv2.VideoCapture(0)

IMG_SIZE = 64

while True:

    ret,frame = cap.read()

    roi = frame[100:400,100:400]

    img = cv2.resize(roi,(IMG_SIZE,IMG_SIZE))
    img = img/255.0
    img = np.reshape(img,(1,IMG_SIZE,IMG_SIZE,3))

    prediction = model.predict(img)

    gesture = gestures[np.argmax(prediction)]

    cv2.rectangle(frame,(100,100),(400,400),(0,255,0),2)

    cv2.putText(frame,gesture,(100,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    cv2.imshow("Gesture Recognition",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()