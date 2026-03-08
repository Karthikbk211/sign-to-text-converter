"""
Step 1 — Data Collection
========================
Shows each letter A-Z one at a time.
Hold up the sign and press SPACE to capture 100 samples.
Press S to skip a letter, Q to quit early.

Saves: data/landmarks.csv
"""

import cv2
import csv
import os
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector

LETTERS     = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
SAMPLES_PER = 100
DATA_DIR    = 'data'
OUTPUT_FILE = os.path.join(DATA_DIR, 'landmarks.csv')

UI = {
    'panel':  (18,  18,  22),
    'border': (55,  55,  65),
    'dim':    (130, 130, 140),
    'accent': (255, 255, 255),
    'good':   (180, 255, 180),
    'warn':   (140, 180, 255),
}

os.makedirs(DATA_DIR, exist_ok=True)


def extract_features(lm_list):
    """
    Normalize landmarks relative to wrist (index 0).
    Returns 42 features: (x, y) for each of 21 landmarks.
    """
    wrist_x, wrist_y = lm_list[0][0], lm_list[0][1]
    features = []
    for lm in lm_list:
        features.append(lm[0] - wrist_x)
        features.append(lm[1] - wrist_y)
    return features


def main():
    cap      = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1, detectionCon=0.8, minTrackCon=0.8)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Check existing data
    existing = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            for row in csv.reader(f):
                if row:
                    existing.add(row[-1])
        print(f'Existing data for: {sorted(existing)}')

    print(f'Collecting {SAMPLES_PER} samples per letter')
    print('SPACE = start capture  |  S = skip  |  Q = quit\n')

    for letter in LETTERS:
        if letter in existing:
            print(f'Skipping {letter} (already have data)')
            continue

        samples   = []
        capturing = False
        print(f'Ready for: {letter}')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            hands, frame = detector.findHands(frame, draw=True, flipType=False)

            hand_detected = len(hands) > 0

            if hand_detected and capturing and len(samples) < SAMPLES_PER:
                lm_list  = hands[0]['lmList']
                features = extract_features(lm_list)
                samples.append(features)

            # ── UI ────────────────────────────────────────────
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), UI['panel'], -1)
            cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
            cv2.line(frame, (0, 80), (w, 80), UI['border'], 1)

            # Letter
            cv2.putText(frame, f'Sign: {letter}', (15, 55),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, UI['accent'], 2)

            # Progress bar
            progress = len(samples)
            bar_w    = 400
            bar_x    = w // 2 - bar_w // 2
            cv2.rectangle(frame, (bar_x, h//2 + 60),
                         (bar_x + bar_w, h//2 + 78), UI['border'], -1)
            if progress > 0:
                fill = int((progress / SAMPLES_PER) * bar_w)
                cv2.rectangle(frame, (bar_x, h//2 + 60),
                             (bar_x + fill, h//2 + 78), UI['good'], -1)
            cv2.putText(frame, f'{progress} / {SAMPLES_PER}',
                        (bar_x + bar_w//2 - 40, h//2 + 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI['accent'], 1)

            # Status text
            if not hand_detected:
                msg, col = 'Show your hand to the camera', UI['warn']
            elif capturing:
                msg, col = f'Capturing "{letter}"...', UI['good']
            else:
                msg, col = f'Hold sign for "{letter}", then press SPACE', UI['dim']

            cv2.putText(frame, msg, (w//2 - 250, h//2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)

            cv2.putText(frame, '[SPACE] Start  [S] Skip  [Q] Quit',
                        (w//2 - 185, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, UI['dim'], 1)

            cv2.imshow('Data Collection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print('Quit.')
                return
            elif key == ord('s'):
                print(f'Skipped {letter}')
                break
            elif key == 32:  # spacebar
                if hand_detected:
                    capturing = True
                    print(f'Capturing {letter}...')

            if len(samples) >= SAMPLES_PER:
                with open(OUTPUT_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for s in samples:
                        writer.writerow(s + [letter])
                print(f'✓ {letter} done — {SAMPLES_PER} samples saved')
                time.sleep(0.4)
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f'\nCollection complete! File: {OUTPUT_FILE}')
    print('Next: python train_sign.py')


if __name__ == '__main__':
    main()
