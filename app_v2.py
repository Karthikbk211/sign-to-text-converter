"""
Step 3 — Live Recognition
==========================
Uses trained model to recognize ASL signs in real time.
Hold a sign steady for 1.5 seconds to add the letter.

Run after train_sign.py is done.
"""

import cv2
import numpy as np
import joblib
import time
from collections import deque, Counter
from cvzone.HandTrackingModule import HandDetector

MODEL_FILE = 'data/model.pkl'

UI = {
    'panel':  (18,  18,  22),
    'border': (55,  55,  65),
    'dim':    (130, 130, 140),
    'accent': (255, 255, 255),
    'good':   (180, 255, 180),
    'warn':   (140, 180, 255),
}


def extract_features(lm_list):
    wrist_x, wrist_y = lm_list[0][0], lm_list[0][1]
    features = []
    for lm in lm_list:
        features.append(lm[0] - wrist_x)
        features.append(lm[1] - wrist_y)
    return features


class Stabilizer:
    """Only output a prediction when same letter dominates last N frames."""
    def __init__(self, window=12, threshold=0.6):
        self.buf       = deque(maxlen=window)
        self.threshold = threshold

    def update(self, pred):
        self.buf.append(pred)
        if not self.buf:
            return None
        top, count = Counter(self.buf).most_common(1)[0]
        if top and count / len(self.buf) >= self.threshold:
            return top
        return None


class WordBuilder:
    """Adds a letter after holding the sign for hold_time seconds."""
    def __init__(self, hold_time=1.5):
        self.hold_time        = hold_time
        self.current_word     = ''
        self.sentence         = []
        self.last_letter      = None
        self.hold_start       = None
        self._last_added_time = 0

    def update(self, letter):
        now = time.time()
        if letter is None:
            self.last_letter = None
            self.hold_start  = None
            return 0.0

        if letter != self.last_letter:
            self.last_letter = letter
            self.hold_start  = now

        elapsed  = now - (self.hold_start or now)
        progress = min(elapsed / self.hold_time, 1.0)

        cooldown_ok = now - self._last_added_time > 1.0
        if elapsed >= self.hold_time and cooldown_ok:
            self.current_word    += letter
            self._last_added_time = now
            self.hold_start       = now
            print(f'Added: {letter} → "{self.current_word}"')

        return progress

    def space(self):
        if self.current_word:
            self.sentence.append(self.current_word)
            self.current_word = ''

    def backspace(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.sentence:
            self.current_word = self.sentence.pop()

    def clear(self):
        self.current_word = ''
        self.sentence     = []

    def get_text(self):
        parts = self.sentence + ([self.current_word] if self.current_word else [])
        return ' '.join(parts)


def draw_hud(frame, letter, conf, hold_progress, wb, fps, hand_detected):
    h, w      = frame.shape[:2]
    panel_w   = 230

    # Top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 60), UI['panel'], -1)
    cv2.addWeighted(ov, 0.92, frame, 0.08, 0, frame)
    cv2.line(frame, (0, 60), (w, 60), UI['border'], 1)
    cv2.putText(frame, 'SIGN LANGUAGE TRANSLATOR', (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, UI['accent'], 1)
    cv2.putText(frame, f'{fps:.0f} fps', (w - panel_w - 70, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, UI['border'], 1)

    # Side panel
    ov2 = frame.copy()
    cv2.rectangle(ov2, (w - panel_w, 60), (w, h), UI['panel'], -1)
    cv2.addWeighted(ov2, 0.92, frame, 0.08, 0, frame)
    cv2.line(frame, (w - panel_w, 60), (w - panel_w, h), UI['border'], 1)

    px = w - panel_w + 14

    # Detected letter
    cv2.putText(frame, 'DETECTED', (px, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI['dim'], 1)
    if letter and hand_detected:
        cv2.putText(frame, letter, (px, 175),
                    cv2.FONT_HERSHEY_DUPLEX, 4.0, UI['accent'], 3)
        cv2.putText(frame, f'{conf*100:.0f}% confidence', (px, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, UI['dim'], 1)
    else:
        cv2.putText(frame, '-', (px, 175),
                    cv2.FONT_HERSHEY_DUPLEX, 4.0, UI['border'], 3)
        cv2.putText(frame, 'No hand detected', (px, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI['warn'], 1)

    cv2.line(frame, (px, 215), (w - 14, 215), UI['border'], 1)

    # Hold progress bar
    cv2.putText(frame, 'HOLD TO ADD', (px, 238),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI['dim'], 1)
    bar_w = w - 14 - px
    cv2.rectangle(frame, (px, 245), (px + bar_w, 258), UI['border'], -1)
    if hold_progress > 0:
        fill = int(hold_progress * bar_w)
        col  = UI['good'] if hold_progress >= 1.0 else UI['accent']
        cv2.rectangle(frame, (px, 245), (px + fill, 258), col, -1)

    cv2.line(frame, (px, 272), (w - 14, 272), UI['border'], 1)

    # Current word
    cv2.putText(frame, 'WORD', (px, 295),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI['dim'], 1)
    cv2.putText(frame, wb.current_word or '-', (px, 335),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, UI['accent'], 2)

    cv2.line(frame, (px, 355), (w - 14, 355), UI['border'], 1)

    # Controls
    for i, ctrl in enumerate(['[SPACE] Space', '[BKSP]  Delete',
                               '[C]     Clear', '[Q]     Quit']):
        cv2.putText(frame, ctrl, (px, 385 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI['border'], 1)

    # Bottom output bar
    ov3 = frame.copy()
    cv2.rectangle(ov3, (0, h - 75), (w - panel_w, h), UI['panel'], -1)
    cv2.addWeighted(ov3, 0.92, frame, 0.08, 0, frame)
    cv2.line(frame, (0, h - 75), (w - panel_w, h - 75), UI['border'], 1)
    cv2.putText(frame, 'OUTPUT', (15, h - 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI['dim'], 1)

    text = wb.get_text()
    if len(text) > 38:
        text = '...' + text[-35:]
    cv2.putText(frame, text or '...', (15, h - 18),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, UI['accent'], 1)

    return frame


def main():
    if not __import__('os').path.exists(MODEL_FILE):
        print(f'ERROR: {MODEL_FILE} not found. Run train_sign.py first!')
        return

    print('Loading model...')
    clf = joblib.load(MODEL_FILE)
    print(f'Classes: {list(clf.classes_)}')

    detector   = HandDetector(maxHands=1, detectionCon=0.8, minTrackCon=0.8)
    stabilizer = Stabilizer(window=12, threshold=0.6)
    wb         = WordBuilder(hold_time=1.5)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    print('\nReady! Hold a sign for 1.5s to add the letter.')
    print('[SPACE] Space  [BACKSPACE] Delete  [C] Clear  [Q] Quit')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        hands, frame = detector.findHands(frame, draw=True, flipType=False)

        letter        = None
        conf          = 0.0
        hold_progress = 0.0
        hand_detected = len(hands) > 0

        if hand_detected:
            lm_list  = hands[0]['lmList']
            features = extract_features(lm_list)
            probs    = clf.predict_proba([features])[0]
            idx      = np.argmax(probs)
            conf     = probs[idx]

            if conf > 0.5:
                letter = clf.classes_[idx]

            stable       = stabilizer.update(letter)
            hold_progress = wb.update(stable)
        else:
            stabilizer.update(None)
            wb.update(None)

        curr_time = time.time()
        fps       = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        frame = draw_hud(frame, letter, conf, hold_progress, wb, fps, hand_detected)
        cv2.imshow('Sign Language Translator', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:
            wb.space()
        elif key == 8:
            wb.backspace()
        elif key == ord('c'):
            wb.clear()

    cap.release()
    cv2.destroyAllWindows()
    print(f'\nFinal: {wb.get_text()}')


if __name__ == '__main__':
    main()
