# Importing all dependent APIs
import os, cv2, random, time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
print("Importing Dependencies Completed.")

# Creating a database for running CNN
dataDir = r"E:\DeepLearning\ImageDatasets\Dataset01-EBI-McD180mlBottom\EBIResult\Images"
categories = ["Pass", "Fail"]
trainingData = []
xTrain = []
yTrain = []
xTest = []
yTest = []
imgSizeC = 256

def createTrainingData():
    for category in categories:
        path = os.path.join(dataDir, category)    # path to cats or dogs dir
        print("Loading " + category + " images.")
        time.sleep(0.25)
        classNum = categories.index(category)
        loop = tqdm(total=len(os.listdir(path)),desc=category, position=0, leave=False)
        for img in os.listdir(path):
            try:
                imgArray = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                cropped = imgArray[:, 88:1112]
                newArrC = cv2.resize(cropped, (imgSizeC, imgSizeC))
                trainingData.append([newArrC, classNum])
                loop.update()
            except Exception as e:
                pass
    random.shuffle(trainingData)

createTrainingData()
print("Length of training data: ",len(trainingData))

for featuresTrain, labelTrain in trainingData[:24000]:
    xTrain.append(featuresTrain)
    yTrain.append(labelTrain)
    
for featuresTest, labelTest in trainingData[24000:]:
    xTest.append(featuresTest)
    yTest.append(labelTest)

xTrain = np.array(xTrain).reshape(-1, imgSizeC, imgSizeC, 1)
xTest = np.array(xTest).reshape(-1, imgSizeC, imgSizeC, 1)

print(xTrain.shape)
print(xTest.shape)
print(yTrain.shape)
print(yTest.shape)

print("Done")

# Pre-processing stage
# Reshaping loaded image dataset to 3 dimensions and plotting sample image.
xTrain = np.array(xTrain, dtype=np.float32)
xTest = np.array(xTest, dtype=np.float32)
xTrain /= 255
xTest /= 255
print("Reshaping done")

# Splitting Train & Test datasets in 2 distinct class labels
# Convert 1-D Class arrays to 2-D Class matrices
yTrain = np_utils.to_categorical(yTrain,2)
yTest = np_utils.to_categorical(yTest,2)

# Declaring sequential model format
model = Sequential()
model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(256,256,1)))
model.add(Convolution2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
print(model.output_shape)

# Model compilation
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Model fitting on Training data
model.fit(xTrain,yTrain,batch_size=32,epochs=10,verbose=1)

# Model evaluation
[loss, accuracy] = model.evaluate(xTest,yTest,verbose=1)
print("Loss: ", loss)
print("Accuracy: ", accuracy*100)
