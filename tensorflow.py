import pickle
import time
X=pickle.load(open('X.pkl','rb'))
y=pickle.load(open('y.pkl','rb'))
X=X/255
y
X.shape
#!pip install keras
#!pip install tensorflow
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.callbacks import TensorBoard
model = Sequential()
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128,input_shape=X.shape[1:],activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
NAME=f'cat-vs-dog-prediction-{int(time.time())}'
tensorboard = TensorBoard(log_dir = 'logs/')
model.fit(X,y,epochs=5,validation_split = 0.1,callbacks = [tensorboard])