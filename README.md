# Photo to Emoji 😄😠😢

Real-time facial emotion recognition that maps your webcam expression to a
matching emoji, using a CNN trained on the **FER-2013** dataset.

<p align="center">
  <i>Webcam feed on the left, predicted emotion + emoji on the right.</i>
</p>

---

## Features

- Convolutional neural network trained from scratch on FER-2013 (7 emotion
  classes: angry, disgust, fear, happy, neutral, sad, surprise)
- Real-time face detection via OpenCV Haar Cascades
- Tkinter GUI showing live webcam feed alongside the matching emoji
- No hardcoded, machine-specific file paths (a common bug in older tutorials
  this project is based on)

## Project structure

```
photo-to-emoji/
├── data/
│   ├── train/              # put FER-2013 training images here
│   └── test/                # put FER-2013 validation/test images here
├── emojis/                  # 7 emoji PNGs (see below)
├── train.py                 # trains the CNN, saves model.h5
├── gui.py                   # runs the live webcam + emoji demo
├── requirements.txt
├── LICENSE
└── README.md
```

Each of `data/train` and `data/test` should contain seven subfolders, one per
emotion, with images inside:

```
data/train/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/photo-to-emoji.git
cd photo-to-emoji
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the dataset

Download FER-2013 (e.g. from Kaggle:
[`fer2013`](https://www.kaggle.com/datasets/msambare/fer2013)) and arrange it
into `data/train/<emotion>/...` and `data/test/<emotion>/...` as shown above.

### 3. Add emoji images

Create/confirm an `emojis/` folder containing 7 PNGs named exactly:

```
angry.png  disgusted.png  fearful.png  happy.png  neutral.png  sad.png  surprised.png
```

Any small square emoji/avatar images work — free sets are easy to find online
(e.g. via [OpenMoji](https://openmoji.org/) or
[Twemoji](https://twemoji.twitter.com/)).

### 4. Train the model

```bash
python train.py
```

This trains for 50 epochs and saves weights to `model.h5` in the project
root. Training on CPU can take a while (hours); a GPU-enabled TensorFlow
install will be much faster. Watch the console output — it prints the
`class_indices` mapping Keras assigned to your folders; this should read
`{'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5,
'surprise': 6}`, which matches the label order hardcoded in `gui.py`. If your
folder names differ, double check this matches before moving on.

### 5. Run the live demo

```bash
python gui.py
```

A window opens with your webcam feed on the left and the predicted
emotion/emoji on the right. Click **Quit** or close the window to exit.

## How it works

1. **Training** (`train.py`): Images are loaded via Keras'
   `ImageDataGenerator`, resized to 48×48 grayscale, and fed into a CNN
   (Conv2D → MaxPooling → Dropout, stacked twice, followed by dense layers
   ending in a 7-way softmax).
2. **Inference** (`gui.py`): OpenCV's Haar Cascade classifier detects face
   bounding boxes in each webcam frame. Each detected face is cropped,
   resized to 48×48, normalized, and passed through the trained model to get
   an emotion prediction, which is then mapped to an emoji image.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Can't open the camera` | Check that no other app is using the webcam, and that you've granted camera permissions to your terminal/IDE. |
| Predictions look index-shifted (e.g. happy shown as sad) | Re-check the `class_indices` printed by `train.py` against `emotion_dict` in `gui.py` — they must match. |
| `model.h5 not found` | Run `train.py` first to generate it. |
| Low accuracy | Train for more epochs, or add data augmentation (rotation/zoom/flip) to the `ImageDataGenerator` calls. |

## Credits

Based on / fixes bugs found in the DataFlair "Emojify" tutorial. This version
corrects several issues present there: a non-existent `keras.emotion_models`
import, an invalid `gray_framescale` color mode, a hardcoded
`/home/<user>/...` haarcascade path, a broken/duplicated-key `emoji_dist`
dictionary, and missing pixel normalization at inference time.

## License

[MIT](LICENSE)
