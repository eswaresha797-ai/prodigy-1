import os
import cv2
import numpy as np
import joblib
import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

IMAGE_SIZE = 64
MODEL_PATH = "svm_cat_dog.pkl"

def load_dataset(data_dir):
    X, y = [], []
    for label, folder in enumerate(["cats", "dogs"]):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            continue
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                X.append(img.flatten())
                y.append(label)
    return np.array(X), np.array(y)

def train_model(data_dir):
    X, y = load_dataset(data_dir)
    if len(X) == 0:
        return None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    joblib.dump(clf, MODEL_PATH)
    return clf, acc

def predict(image):
    clf = joblib.load(MODEL_PATH)
    img = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    img_flat = img.flatten().reshape(1, -1)
    pred = clf.predict(img_flat)[0]
    prob = clf.predict_proba(img_flat)[0]
    return ("Cat" if pred == 0 else "Dog"), prob

st.title("🐱 Cat vs Dog Classifier (SVM)")

menu = ["Train Model", "Test Image"]
choice = st.sidebar.selectbox("Select Mode", menu)

if choice == "Train Model":
    st.subheader("📂 Train the SVM Model")
    data_dir = st.text_input("Enter dataset folder path (must contain 'cats' and 'dogs' subfolders):")
    if st.button("Train"):
        if os.path.exists(data_dir):
            clf, acc = train_model(data_dir)
            if clf is not None:
                st.success(f"✅ Model trained successfully! Accuracy = {acc*100:.2f}%")
            else:
                st.error("Dataset not found or empty!")
        else:
            st.error("❌ Invalid dataset path!")

elif choice == "Test Image":
    st.subheader("📷 Upload an Image for Prediction")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)
        if st.button("Predict"):
            if os.path.exists(MODEL_PATH):
                label, prob = predict(image)
                st.success(f"Prediction: *{label}*")
                st.write(f"Confidence: {max(prob)*100:.2f}%")
            else:
                st.error("Model not trained yet. Go to 'Train Model' first.")
