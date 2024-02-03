import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

###################### read input data and preprocess ##############################
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Setup the train and test directories
train_dir = "dataset/Training/"
valid_dir = "dataset/Validation/"

# Import data from directories and turn it into batches
train_data = train_datagen.flow_from_directory(train_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)

valid_data = valid_datagen.flow_from_directory(valid_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)

###################### model definition ##############################
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64,
                           kernel_size=3,
                           activation='relu',
                           input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64,
                           activation='relu',
                           kernel_size=3),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64,
                           activation='relu',
                           kernel_size=3),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

###################### training ##############################

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="models/best_model.h5",
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1)

class earlyStoppingCallback(tf.keras.callbacks.Callback):
    # Define the correct function signature for on_epoch_end
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.9:
            print("\nReached 75% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = [earlyStoppingCallback(), model_checkpoint_callback]

model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# Fit the model with the TensorBoard callback
history_1 = model.fit(train_data,
                      epochs=50,
                      steps_per_epoch=len(train_data),
                      validation_data=valid_data,
                      callbacks=callbacks,
                      validation_steps=len(valid_data))

model.save_weights("models/after_training.h5")

