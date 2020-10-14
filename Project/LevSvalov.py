#imports
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, Nadam, RMSprop
from tensorflow.keras.layers import Conv2D, MaxPool2D,Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import keras.backend as K
import numpy as np

x_train = np.load('/Users/levsvalov/code_workspace/Fall2020/ML/Project/input_data/x.npy')
y_train = np.load('/Users/levsvalov/code_workspace/Fall2020/ML/Project/input_data/y.npy')

def make_model():
    m = Sequential()
    m.add(Conv2D(8,kernel_size=5,input_shape=(28,28,1),activation='relu'))
    m.add(MaxPool2D(pool_size=2,strides=1))
    m.add(Conv2D(19,kernel_size=3,input_shape=(28,28,1),activation='relu')) # ooo santiii cazooorla
    m.add(MaxPool2D(pool_size=2))
    m.add(Conv2D(34,kernel_size=3,input_shape=(28,28,1),activation='relu')) # ee granit xhaka
    m.add(MaxPool2D(pool_size=2,strides=1))
    m.add(BatchNormalization())
    m.add(Flatten())
    m.add(Dense(71,activation='relu')) # kovi
    m.add(BatchNormalization())
    m.add(Dropout(0.2))
    m.add(Dense(130,activation='relu'))
    m.add(BatchNormalization())
    m.add(Dropout(0.2))
    m.add(Dense(77,activation='relu')) # ex num for young talented guy - bakayko saka
    m.add(BatchNormalization())
    m.add(Dropout(0.2))
    m.add(Dense(10,activation='softmax'))
    return m

model = make_model()

generator = ImageDataGenerator(zca_whitening=True,rotation_range=35, width_shift_range=0.15,height_shift_range=0.15,brightness_range=[0.2,0.8],zoom_range=0.25,validation_split=0.15)
lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy',factor=0.25,patience=8)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
model.compile(optimizer=Nadam(1e-3,clipnorm=1),loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(generator.flow(x_train,y_train, batch_size=128),
          validation_data=generator.flow(x_train,y_train, batch_size=128,subset='validation'),
          epochs=27,verbose=2,callbacks=[early_stopping,lr_scheduler],
          steps_per_epoch=(len(x_train)/128))

# model.save('/Users/levsvalov/code_workspace/Fall2020/ML/Project/model.h5')
output = os.getcwd() + "/model.h5"
model.save(output)
print("done")