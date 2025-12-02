import cv2
import numpy as np
from skimage.feature import hog
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def extract_features(folder):
    features = []
    labels = []

    for label, subfolder in enumerate(['human', 'non_human']):
        path = os.path.join(folder, subfolder)

        for file in os.listdir(path):
            if file.startswith('.'):
                continue

            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (64, 128))   
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            hog_features = hog(gray,
                               orientations=9,
                               pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               block_norm='L2-Hys')

            features.append(hog_features)
            labels.append(label)

    return np.array(features), np.array(labels)


DATASET_PATH = '/Users/bhumikaburhade/Desktop/dataset'

print("\n--- TRAINING MODEL ---")

features, labels = extract_features(DATASET_PATH)
print("Features shape:", features.shape)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

clf = LinearSVC()
clf.fit(X_train, y_train)

accuracy = accuracy_score(y_test, clf.predict(X_test))
print("Model Accuracy:", round(accuracy*100, 2), "%")


cap = cv2.VideoCapture(0)
print("\nPress Q to quit webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (64, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    hog_features = hog(gray,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       block_norm='L2-Hys')

    hog_features = hog_features.reshape(1, -1)

    confidence = clf.decision_function(hog_features)[0]
    confidence_score = abs(confidence)
    pred = clf.predict(hog_features)[0]


    if pred == 0:
        label_text = "Human Detected"
        color = (0, 255, 0)
    else:
        label_text = "No Human"
        color = (0, 0, 255)


    cv2.putText(frame, label_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, f"Confidence: {round(confidence_score, 2)}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Human Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()