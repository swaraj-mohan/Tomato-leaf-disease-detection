#import the required libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from glob import glob

#resize all images to the expected size
image_size = [224, 224]

#set train, test, validation dataset path
train_path = '/content/drive/My Drive/Datasets/Tomato Leaf Disease Prediction/New Plant Diseases Dataset(Augmented)/train'
valid_path = '/content/drive/My Drive/Datasets/Tomato Leaf Disease Prediction/New Plant Diseases Dataset(Augmented)/valid'

#import the InceptionV3 architecture and add preprocessing layer, we are using ImageNet weights
InceptionV3_model = keras.applications.inception_v3.InceptionV3(input_shape = image_size + [3], weights = 'imagenet', include_top = False)

#freeze the weights of the pre-trained layers
for layer in InceptionV3_model.layers:
  layer.trainable = False
  
#useful for getting number of output classes
folders = glob('/content/drive/My Drive/Datasets/Tomato Leaf Disease Prediction/New Plant Diseases Dataset(Augmented)/train/*')


#adding our own layers
layer_flatten = keras.layers.Flatten()(InceptionV3_model.output)
output = keras.layers.Dense(len(folders), activation = "softmax")(layer_flatten)
model = keras.Model(inputs = InceptionV3_model.input, outputs = output)

print(model.summary())

#compile the model and specify loss function and optimizer
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

#use the ImageDataGenerator class to load images from the dataset
train_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   validation_split=0.2)

#make sure you provide the same target size as initialied for the image size
training_set = train_data_generator.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 subset='training')

validation_set = train_data_generator.flow_from_directory(train_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical',
                                            subset='validation')

#train the model
history = model.fit_generator(
  training_set,
  validation_data = validation_set,
  epochs = 10,
  steps_per_epoch = len(training_set),
  validation_steps = len(validation_set)
)

#save the model as an h5 file
model.save('/content/drive/My Drive/Datasets/Tomato Leaf Disease Prediction/model_InceptionV3.h5')

#plot the loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

#plot the accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

print(model.evaluate(validation_set))

#using the model to make predictions
y_pred = model.predict(validation_set)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)