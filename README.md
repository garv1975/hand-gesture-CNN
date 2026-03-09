# Hand Gesture Recognition using CNN

## 📌 Project Overview

This project implements a **Hand Gesture Recognition System** using **Convolutional Neural Networks (CNN)** and the **LeapGestRecog dataset**.
The model is trained to classify **10 different hand gestures** and can later be used for real-time gesture recognition using a webcam.

Hand gesture recognition is useful in **human–computer interaction, virtual reality, sign language interpretation, and touchless interfaces**.

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn

---

## 📂 Dataset

This project uses the **LeapGestRecog Dataset**, which contains **20,000 hand gesture images** across **10 gesture classes**.

Dataset structure:

```
LeapGestRecog
│
├── 00
│   ├── 01_palm
│   ├── 02_l
│   ├── 03_fist
│   ├── 04_fist_moved
│   ├── 05_thumb
│   ├── 06_index
│   ├── 07_ok
│   ├── 08_palm_moved
│   ├── 09_c
│   └── 10_down
│
├── 01
├── 02
...
```

Each folder represents a different user performing the gestures.

---

## ✋ Gesture Classes

The model classifies the following gestures:

1. Palm
2. L
3. Fist
4. Fist Moved
5. Thumb
6. Index
7. OK
8. Palm Moved
9. C
10. Down

---

## ⚙️ Installation

Install the required dependencies:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

---

## 🚀 How to Run the Project

### Step 1 — Download Dataset

Download the **LeapGestRecog dataset** and place it inside the project folder.

Example structure:

```
hand_gesture_CNN
│
├── gesture_train.py
├── gesture_webcam.py
└── LeapGestRecog
```

---

### Step 2 — Train the Model

Run the training script:

```bash
python gesture_train.py
```

The model will train for several epochs and generate a trained model file:

```
gesture_model.h5
```

---

### Step 3 — Run Real-Time Gesture Detection

After training, run:

```bash
python gesture_webcam.py
```

The webcam will start and detect hand gestures in real time.

---

## 🧩 Model Architecture

The CNN model consists of:

* 3 Convolutional Layers
* 3 MaxPooling Layers
* Flatten Layer
* Dense Fully Connected Layer
* Dropout Layer (to prevent overfitting)
* Output Layer with Softmax activation

Input image size: **64 × 64**

---

## 📊 Training Output

During training the system prints:

* Training samples
* Testing samples
* Accuracy
* Validation accuracy

Typical results:

```
Training Accuracy: 95%+
Validation Accuracy: 94%+
```

---

## 📈 Output Visualization

The training script plots a graph showing:

* Training Accuracy
* Validation Accuracy

This helps visualize model performance.

---

## 📁 Project Structure

```
hand_gesture_CNN
│
├── gesture_train.py
├── gesture_webcam.py
├── gesture_model.h5
├── README.md
└── LeapGestRecog/
```

---

## 🔮 Future Improvements

Possible improvements include:

* Using **MediaPipe for better hand tracking**
* Increasing model accuracy using **Transfer Learning**
* Building a **GUI interface**
* Implementing **Sign Language Recognition**
* Deploying the system as a **web application**

---

## 👨‍💻 Author

Developed as part of a **Machine Learning / Computer Vision project** for gesture recognition.

---

## 📜 License

This project is for **educational and research purposes**.
