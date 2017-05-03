from sklearn import model_selection
# from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
import os, glob
import numpy as np

root_dir = "./images/"
categories = ['hamajima', 'nozawa', 'imai']
nb_classes = len(categories)
image_size = 32

X = []
Y = []
for idx, cat in enumerate(categories):
    files = glob.glob(root_dir + "/" + cat + "/*")
    print("---", cat, "を処理中")
    for i, f in enumerate(files):
        img = load_img(f, target_size=(image_size,image_size))
        data = img_to_array(img)
        X.append(data)
        Y.append(idx)
X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./images/face.npy", xy)
print("ok,", len(Y))