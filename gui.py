"""
gui.py
Opens your webcam, detects faces, predicts the emotion using the model
trained by train.py, and shows the matching emoji side-by-side.

Folder structure expected next to this file:
    model.h5                  <- produced by train.py
    emojis/
        angry.png
        disgusted.png
        fearful.png
        happy.png
        neutral.png
        sad.png
        surprised.png

Run:
    python gui.py
"""

import tkinter as tk
from tkinter import Label, Button

import cv2
import numpy as np
from PIL import Image, ImageTk

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# ---------------------------------------------------------------------------
# 1. Rebuild the model architecture (must match train.py exactly) and load
#    the trained weights.
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

emotion_model.load_weights("model.h5")

cv2.ocl.setUseOpenCL(False)

# ---------------------------------------------------------------------------
# 2. Label <-> emoji mapping
#    NOTE: this order MUST match the class_indices printed by train.py
#    (Keras assigns indices alphabetically by folder name, which gives this
#    order for: angry, disgust, fear, happy, neutral, sad, surprise)
# ---------------------------------------------------------------------------
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}

emoji_dist = {
    0: "./emojis/angry.png",
    1: "./emojis/disgusted.png",
    2: "./emojis/fearful.png",
    3: "./emojis/happy.png",
    4: "./emojis/neutral.png",
    5: "./emojis/sad.png",
    6: "./emojis/surprised.png",
}

# Path to the haarcascade file that ships with opencv-python.
# Using cv2.data.haarcascades avoids hardcoding a user-specific path.
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
bounding_box = cv2.CascadeClassifier(CASCADE_PATH)

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]


def show_vid():
    global cap1
    cap1 = cv2.VideoCapture(0)
    if not cap1.isOpened():
        print("Can't open the camera")
        return

    flag1, frame1 = cap1.read()
    if flag1 is None or frame1 is None:
        lmain.after(10, show_vid)
        return

    frame1 = cv2.resize(frame1, (600, 500))

    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        cropped_img = cropped_img.astype("float32") / 255.0

        prediction = emotion_model.predict(cropped_img, verbose=0)
        maxindex = int(np.argmax(prediction))
        cv2.putText(
            frame1,
            emotion_dict[maxindex],
            (x + 20, y - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        show_text[0] = maxindex

    if flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

    lmain.after(10, show_vid)


def show_vid2():
    frame2 = cv2.imread(emoji_dist[show_text[0]])
    if frame2 is None:
        # Emoji image missing - skip this refresh instead of crashing.
        lmain2.after(10, show_vid2)
        return

    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(pic2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2 = imgtk2
    lmain2.configure(image=imgtk2)
    lmain3.configure(text=emotion_dict[show_text[0]], font=("arial", 45, "bold"))

    lmain2.after(10, show_vid2)


def on_close():
    try:
        if cap1 is not None:
            cap1.release()
    except NameError:
        pass
    root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")
    root["bg"] = "black"

    heading2 = Label(
        root, text="Photo to Emoji", pady=20, font=("arial", 45, "bold"), bg="black", fg="#CDCDCD"
    )
    heading2.pack()

    lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=10)
    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg="black")

    lmain.pack(side=tk.LEFT)
    lmain.place(x=50, y=250)

    lmain3.pack()
    lmain3.place(x=960, y=250)

    lmain2.pack(side=tk.RIGHT)
    lmain2.place(x=900, y=350)

    exitbutton = Button(
        root, text="Quit", fg="red", command=on_close, font=("arial", 25, "bold")
    )
    exitbutton.pack(side=tk.BOTTOM)

    root.protocol("WM_DELETE_WINDOW", on_close)

    show_vid()
    show_vid2()
    root.mainloop()
