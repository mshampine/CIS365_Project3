from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import os

"""
Project 3 : CIS365-01
Title: Gesture ANN Camera
Description:
Date: 2/29/2020
Authors: Matt Shampine, Nabeel Vali

This file hold the model and its training code
"""

# Establish the base directory location for all the images
base_dir = './Images'

# Append the test and training directory paths to the base directory
train_dir = os.path.join(base_dir, 'Train')
test_dir = os.path.join(base_dir, 'Test')

# Append each classes training path to the base training path
train_FlatHand_dir = os.path.join(train_dir, "FlatHand")
#train_Fist_dir = os.path.join(train_dir, "Fist" )
train_Thumbs_dir = os.path.join(train_dir, "Thumbs")
train_BlankWall_dir = os.path.join(train_dir, "BlankWall")

# Append each classes testing path to the base test path
test_FlatHand_dir = os.path.join(test_dir, "FlatHand")
#test_Fist_dir = os.path.join(test_dir, "Fist")
test_Thumbs_dir = os.path.join(test_dir, "Thumbs")
test_BlankWall_dir = os.path.join(test_dir, "BlankWall")


# Rescale our data from 0 - 255 to 0 - 1, for use in the model.
train_datagen = ImageDataGenerator( rescale = 1.0/255.)
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Generate batches of training and testing data from the batches created above
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=10,
                                                    class_mode='categorical',
                                                    target_size=(224,224),
                                                    shuffle=True)

test_generator = test_datagen.flow_from_directory(test_dir,
                                                    batch_size=10,
                                                    class_mode='categorical',
                                                    target_size=(224,224),
                                                    shuffle=True)


# Build a CNN model by sequentially adding the various hidden layers.
# Convolution layers build a feature map out of our image input, it also scales the image down
# Max pooling layers further downsample our image input
# Dropout layers, with a certain probability, discard some sample data to prevent training on noise
# Flatten() compresses an input before feeding it into the dense layer
# Dense layers are used to aggregate the model data and form a prediction
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.Dense(3, activation="softmax")
])

# Optimizer used: Stochastic Gradient Descent
# We set a higher momentum to increase the likelihood of convergence on an optimal point
optimizer = SGD(lr=0.001, momentum=0.9)

model.compile(optimizer=optimizer,
               loss="categorical_crossentropy",
               metrics=['accuracy'])

# Train the model using the training data, validate it with our test data
model.fit(train_generator,
      shuffle=True,
      epochs=3,
      validation_data=test_generator,
      verbose=1,
)

model.save('finalModel.h5')

