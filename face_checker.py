import face_keras as face
import sys, os
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
import cv2
import time
import keras.backend as K


cascade_path = "C:\Tools\opencv\sources\data\haarcascades_cuda\haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
color = (255, 255, 255)

image_size = 32

def facedetect(file):
    K.clear_session()
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.07, minNeighbors=9, minSize=(10, 10))
    print(len(faces))
    i = 1
    for rect in faces:
        BORDER_COLOR = (255, 255, 255)
        cv2.rectangle(
            img,
            tuple(rect[0:2]),
            tuple(rect[0:2] + rect[2:4]),
            BORDER_COLOR,
            thickness=2
        )
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        font = cv2.FONT_HERSHEY_SIMPLEX

        dst = img[y:y+h, x:x+w]
        cv2.imwrite('./output.jpg', dst)

        face_img = load_img('./output.jpg', target_size=(image_size, image_size))
        in_data = img_to_array(face_img)

        X = []
        X.append(in_data)
        X = np.array(X)
        X = X.astype('float') / 256

        model = face.build_model(X.shape[1:])
        model.load_weights('./images/face-model.h5')

        pre = model.predict(X)
        print(pre)

        color = (255, 255, 255)
        pre_str = 'unknown'
        if pre[0][0] > 0.5:
            pre_str = "hamajima(%d)" % round(pre[0][0] * 100, 3)
            color = (255, 255, 0)
        elif pre[0][1] > 0.5:
            pre_str = "nozawa(%d)" % round(pre[0][1] * 100, 3)
            color = (255, 0, 255)
        elif pre[0][1] > 0.5:
            pre_str = "imai(%d)" % round(pre[0][2] * 100, 3)
            color = (0, 255, 255)

        cv2.putText(img, ("%d %s" % (i, pre_str)), (x, y), font, 0.5, color)

        i = i + 1

    name, ext = os.path.splitext(os.path.basename(file))
    cv2.imwrite(name + "_result.jpg", img)

if __name__ == '__main__':
    image = sys.argv[1]
    if os.path.exists(image):
        facedetect(image)
