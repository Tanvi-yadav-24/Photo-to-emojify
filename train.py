"""
train.py
Trains a CNN on the FER-2013 dataset to classify facial emotions,
then saves the weights to model.h5 (used later by gui.py).

Expected folder structure (matches the FER dataset folders you have):

FER dataset/
    train/
        angry/
        disgust/
        fear/
        happy/
        neutral/
        sad/
        surprise/
    test/
        angry/
        disgust/
        fear/
        happy/
        neutral/
        sad/
        surprise/

Run:
    python train.py
"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# 1. Paths - EDIT these two lines to point at your actual dataset folder
# ---------------------------------------------------------------------------
train_dir = "data/train"
val_dir = "data/test"

# ---------------------------------------------------------------------------
# 2. Data generators
# ---------------------------------------------------------------------------
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
)

# IMPORTANT: this tells us the order Keras assigned to the emotion folders.
# We need this exact order later when mapping predictions -> emoji.
print("Class indices (folder -> label index):")
print(train_generator.class_indices)

num_train_imgs = train_generator.samples
num_val_imgs = validation_generator.samples

# ---------------------------------------------------------------------------
# 3. Model architecture
# ---------------------------------------------------------------------------
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation="relu"))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation="softmax"))

emotion_model.summary()

# ---------------------------------------------------------------------------
# 4. Compile and train
# ---------------------------------------------------------------------------
try:
    optimizer = Adam(learning_rate=0.0001, weight_decay=1e-6)
except TypeError:
    # older keras versions use 'lr' and 'decay' instead
    optimizer = Adam(lr=0.0001, decay=1e-6)

emotion_model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)

EPOCHS = 50
BATCH_SIZE = 64

emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=num_train_imgs // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=num_val_imgs // BATCH_SIZE,
)

# ---------------------------------------------------------------------------
# 5. Save weights
# ---------------------------------------------------------------------------
emotion_model.save_weights("model.h5")
print("Saved trained weights to model.h5")
