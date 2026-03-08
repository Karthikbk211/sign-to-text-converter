# 🤟 Sign Language Translator

A real-time ASL (American Sign Language) letter recognition system that translates hand signs into text using computer vision and machine learning.

---

## 🧠 How It Works

```
Webcam → CVZone Hand Detection → 21 Landmarks → 
Normalized Features → Random Forest → Letter → Word Builder
```

The model is trained on **your own hand** — you collect the data, train the classifier, and run live recognition. This makes it personalized and more accurate than generic pretrained models.

**Feature extraction:**
- CVZone detects 21 hand landmarks per frame
- Landmarks are normalized relative to wrist position (translation-invariant)
- 42 features total (x, y for each landmark)
- Random Forest classifier with 200 trees

---

## 🚀 Getting Started

```bash
git clone https://github.com/YOURUSERNAME/sign-language-translator
cd sign-language-translator
pip install -r requirements.txt
```

### Step 1 — Collect Data
```bash
python collect_v2.py
```
Goes through each letter A-Z. Hold up the sign and press `SPACE` to capture 100 samples. Press `S` to skip, `Q` to quit — progress is saved automatically.

### Step 2 — Train
```bash
python train_sign.py
```
Trains a Random Forest classifier on your data. Takes ~10 seconds. Outputs a confusion matrix and saves the model.

### Step 3 — Live Recognition
```bash
python app_v2.py
```
Hold a sign steady for 1.5 seconds to add the letter to the output.

---

## 🎮 Controls (app_v2.py)

| Key | Action |
|---|---|
| Hold sign 1.5s | Add letter |
| `SPACE` | Add space |
| `BACKSPACE` | Delete last letter |
| `C` | Clear all |
| `Q` | Quit |

---

## 📁 Project Structure

```
sign-language-translator/
├── collect_v2.py       # Step 1 — data collection
├── train_sign.py       # Step 2 — train classifier
├── app_v2.py           # Step 3 — live recognition
├── data/
│   ├── landmarks.csv   # collected landmark data
│   ├── model.pkl       # trained Random Forest
│   └── confusion_matrix.png
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```
cvzone>=1.5.6
opencv-python>=4.7.0
mediapipe==0.10.14
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.2.0
```

---

## 💡 Key Design Decisions

- **Personalized model** — trained on your own hand so it adapts to your signing style
- **Wrist-relative normalization** — features are translation-invariant so position on screen doesn't matter
- **Prediction stabilizer** — requires same letter in 60% of last 12 frames before accepting, prevents flickering
- **Hold-to-add** — 1.5 second hold prevents accidental letter additions
- **CVZone** — cleaner hand tracking API built on MediaPipe

---

## 👤 Author

**Karthik Bhaskar Kamuju** — [GitHub](https://github.com/Karthikbk211) · [LinkedIn](https://www.linkedin.com/in/Karthik-Kamuju)
