"""
Step 2 — Train Classifier
==========================
Trains a Random Forest on the collected landmark data.
Takes ~10 seconds. Saves model to data/model.pkl

Run after collect.py is done.
"""

import numpy as np
import csv
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE  = 'data/landmarks.csv'
MODEL_FILE = 'data/model.pkl'


def load_data(path):
    X, y = [], []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            X.append([float(v) for v in row[:-1]])
            y.append(row[-1])
    return np.array(X), np.array(y)


def main():
    if not os.path.exists(DATA_FILE):
        print(f'ERROR: {DATA_FILE} not found. Run collect.py first!')
        return

    print('Loading data...')
    X, y = load_data(DATA_FILE)

    labels = sorted(set(y))
    print(f'Classes: {labels}')
    print(f'Total samples: {len(X)}')
    print(f'Samples per class: {len(X) // len(labels)} avg\n')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print('Training Random Forest...')
    clf = RandomForestClassifier(
        n_estimators = 200,
        max_depth    = 20,
        random_state = 42,
        n_jobs       = -1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = (y_pred == y_test).mean() * 100

    print(f'\nTest Accuracy: {acc:.2f}%')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5)
    plt.title(f'Confusion Matrix — Test Accuracy: {acc:.2f}%',
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('data/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: data/confusion_matrix.png')

    joblib.dump(clf, MODEL_FILE)
    print(f'\nModel saved to {MODEL_FILE}')
    print('Now run: python app.py')


if __name__ == '__main__':
    main()
