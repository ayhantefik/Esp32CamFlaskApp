import tensorflow as tf
from tensorflow import keras
import numpy as np

TRAINING_DIR = "images/train/"
TEST_DIR = "images/test/"
VALIDATION_DIR = "images/validation/"
REAL_IMAGES_DIR = "images/real/"

# TRAINING

# training_data = keras.utils.image_dataset_from_directory(
#     TRAINING_DIR, image_size=(320, 240)
# )

# test_data = keras.utils.image_dataset_from_directory(
#     TEST_DIR, image_size=(320, 240)
# )

# validation_data = keras.utils.image_dataset_from_directory(
#     VALIDATION_DIR, image_size=(320, 240),
#     label_mode=None
# )

# model = keras.Sequential([
#     keras.layers.Input(shape=(320, 240, 3)),
#     keras.layers.Rescaling(1./255),
#     keras.layers.RandomRotation(0.4),
#     keras.layers.RandomZoom(0.2),
#     keras.layers.RandomFlip("horizontal"),
#     keras.layers.Conv2D(64, (3,3), activation='relu',
#                         input_shape=(320, 240, 3)),
#     keras.layers.MaxPooling2D(2, 2),
#     keras.layers.Conv2D(64, (3,3), activation='relu'),
#     keras.layers.MaxPooling2D(2, 2),
#     keras.layers.Conv2D(128, (3,3), activation='relu'),
#     keras.layers.MaxPooling2D(2, 2),
#     keras.layers.Conv2D(128, (3,3), activation='relu'),
#     keras.layers.MaxPooling2D(2, 2),
#     keras.layers.Flatten(),  # Flatten 2D image into 1D
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(512, activation='relu'),  # Hidden layer with 128 neurons
#     keras.layers.Dense(7, activation='softmax')  
# ])

# model.compile(loss ='sparse_categorical_crossentropy',
#               optimizer='rmsprop', metrics=['accuracy'])

# history = model.fit(training_data, epochs=20,
#                             validation_data = test_data)

# model.save("my_fingercount_model.keras")

# model = keras.models.load_model("my_model.keras")

# img = keras.utils.load_img(
# REAL_IMAGES_DIR + "four.jpg",
# target_size=(320, 240)
# )

# x = keras.utils.img_to_array(img)
# x = np.expand_dims(x, axis=0)


# prediction = np.argmax(model.predict(x))
# match prediction:
#     case 0:
#         print("Noll")
#     case 1:
#         print("One")
#     case 2:
#         print("Two")
#     case 3:
#         print("Three")
#     case 4:
#         print("Four")
#     case 5:
#         print("Five")
#     case 6:
#         print("Emptyyyy")

# # # ## # #

def get_model_result():
    model = keras.models.load_model("my_fingercount_model.keras")

    img = keras.utils.load_img("image.jpg", target_size=(320, 240))

    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    prediction = np.argmax(model.predict(x))
    
    match prediction:
        case 0:
            return "Zero"
        case 1:
            return "One"
        case 2:
            return "Two"
        case 3:
            return "Three"
        case 4:
            return "Four"
        case 5:
            return "Five"
        case 6:
            return " "