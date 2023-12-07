import numpy as np
from keras.datasets import mnist
import keras
from keras.utils import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import cv2
import os

IMAGE_SIZE = (28,28)
train_path = './Dataset/train'
x_train=[]

for folder in os.listdir(train_path):
    sub_path=train_path+"/"+folder
    for img in os.listdir(sub_path):
        image_path=sub_path+"/"+img
        img_arr=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        img_arr=cv2.resize(img_arr,(28,28))
        x_train.append(img_arr)

val_path = './Dataset/val'
x_val=[]

for folder in os.listdir(val_path):
    sub_path=val_path+"/"+folder
    for img in os.listdir(sub_path):
        image_path=sub_path+"/"+img
        img_arr=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        img_arr=cv2.resize(img_arr,(28,28))
        x_val.append(img_arr)

train_x=np.array(x_train)
val_x=np.array(x_val)
train_x=train_x/255.0
val_x=val_x/255.0
train_x = np.expand_dims(train_x,-1)
val_x = np.expand_dims(val_x,-1)

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (28, 28),
                                                 batch_size = 32,
                                                 class_mode = 'sparse')

val_set = train_datagen.flow_from_directory(val_path,
                                                 target_size = (28, 28),
                                                 batch_size = 32,
                                                 class_mode = 'sparse')

train_y = to_categorical(training_set.classes,num_classes=9)
val_y = to_categorical(val_set.classes,num_classes=9)


num_classes = 9

model = keras.Sequential(
    [
        keras.Input(shape=(28,28,1)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

batch_size = 32
epochs = 100
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(train_x,train_y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
score = model.evaluate(val_x,val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
model.save('captcha.h5')