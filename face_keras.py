from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import EarlyStopping
from keras.optimizers import rmsprop
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import glob
from sklearn.model_selection import train_test_split

root_dir = "./images/"
categories = ['hamajima', 'nozawa', 'imai']
nb_classes = len(categories)
image_size = 32


def main():
    K.clear_session()
    x_train, x_test, y_train, y_test = load_image()
    x_train = x_train.astype("float") / 256
    x_test = x_test.astype("float") / 256
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    model = model_train(x_train, y_train, x_test, y_test)
    model_eval(model, x_test, y_test)


def load_image():
    x = []
    y = []
    for idx, cat in enumerate(categories):
        files = glob.glob(root_dir + "/" + cat + "/*")
        print("---", cat, "を処理中")
        for i, f in enumerate(files):
            img = load_img(f, target_size=(image_size, image_size))
            data = img_to_array(img)
            x.append(data)
            y.append(idx)
    x = np.array(x)
    y = np.array(y)

    return train_test_split(x, y)


def build_model(in_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def model_train(x_train, y_train, x_test, y_test):
    model = build_model(x_train.shape[1:])

    print(model.summary())

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False)
    datagen.fit(x_train)

    early_stop = EarlyStopping(patience=100)
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                                  steps_per_epoch=2000,
                                  epochs=20,
                                  validation_data=datagen.flow(x_test, y_test, batch_size=32),
                                  validation_steps=800,
                                  callbacks=[early_stop])

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    hdf5_file = "./images/face-model.h5"
    model.save_weights(hdf5_file)
    return model


def model_eval(model, X, y):
    score = model.evaluate(X, y)
    print('loss=', score[0])
    print('accuracy=', score[1])


if __name__ == "__main__":
    main()
