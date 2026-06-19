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
│   ├── train/                       # empty on purpose — see "About data/" below
│   └── test/                        # empty on purpose — see "About data/" below
├── emojis/                          # 7 emoji PNGs (see below)
├── train.py                         # trains the CNN locally, saves model.h5
├── train_emotion_model.ipynb        # same training, GPU-ready for Colab/Kaggle
├── gui.py                           # runs the live webcam + emoji demo
├── requirements.txt
├── LICENSE
└── README.md
```

> **About `data/`:** these folders are intentionally empty in this repo (just
> placeholders so the expected structure is visible). The FER-2013 dataset is
> ~28,000+ small images — too large/numerous to upload through GitHub's web
> UI or to vendor inside a git repo. **Do not try to upload the dataset to
> GitHub.** Instead, download it locally (or directly inside a Kaggle/Colab
> notebook) using the instructions below.
>
> Likewise, the trained `model.h5` file is **not** committed to this repo —
> it's a generated artifact. You produce your own by running `train.py` or
> the notebook, then keep it locally next to `gui.py`.

Each of `data/train` and `data/test`, once populated, should contain seven
subfolders, one per emotion, with images inside:

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

You have two options — **don't put the dataset in this git repo either way**:

**Option A — Train in the cloud (recommended, faster, no local download needed)**
Open `train_emotion_model.ipynb` in [Google Colab](https://colab.research.google.com/)
or [Kaggle Notebooks](https://www.kaggle.com/code) (enable the free GPU in
both). It downloads FER-2013 itself via `kagglehub` and trains on a GPU —
typically 15–30 minutes instead of several hours on a laptop CPU. At the
end, download just the resulting `model.h5` and skip to step 5 below.

**Option B — Train locally**
Download FER-2013 (e.g. from Kaggle:
[`fer2013`](https://www.kaggle.com/datasets/msambare/fer2013)) to your own
machine and arrange it into `data/train/<emotion>/...` and
`data/test/<emotion>/...` as shown above. This folder stays local only —
it's already excluded via `.gitignore` so it won't get committed.

### 3. Add emoji images

Create/confirm an `emojis/` folder containing 7 PNGs named exactly:

```
angry.png  disgusted.png  fearful.png  happy.png  neutral.png  sad.png  surprised.png
```

Any small square emoji/avatar images work — free sets are easy to find online
(e.g. via [OpenMoji](https://openmoji.org/) or
[Twemoji](https://twemoji.twitter.com/)).

### 4. Train the model (skip if you already did this in Colab/Kaggle)

```bash
python train.py
```

This trains for 50 epochs and saves weights to `model.h5` in the project
root. Training on CPU can take a while (hours) — see Option A above if you'd
rather use a free GPU instead. Watch the console output — it prints the
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
| `model.h5 not found` | Run `train.py` (or the notebook) first to generate it, then make sure it's sitting next to `gui.py`. |
| GitHub upload of `data/` "didn't work" / folder looks empty | Expected — see "About `data/`" note above. The dataset is intentionally never stored in this repo; download it locally or use the notebook instead. |
| Low accuracy | Train for more epochs, or add data augmentation (rotation/zoom/flip) to the `ImageDataGenerator` calls. |

## Credits

Based on / fixes bugs found in the DataFlair "Emojify" tutorial. This version
corrects several issues present there: a non-existent `keras.emotion_models`
import, an invalid `gray_framescale` color mode, a hardcoded
`/home/<user>/...` haarcascade path, a broken/duplicated-key `emoji_dist`
dictionary, and missing pixel normalization at inference time.

## License

[MIT](LICENSE)
