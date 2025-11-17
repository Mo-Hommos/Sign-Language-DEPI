# ✋🤖 Arabic Sign Language Translator — DEPI Project  
Real-time Arabic Sign Language → Text Translation using MediaPipe + Machine Learning + Streamlit

## 🚀 Overview  
This project delivers a full end-to-end real-time system that allows Arabic Sign Language (ArSL) users to communicate effortlessly through AI-powered sign recognition.  
The app uses a webcam feed, extracts hand landmarks using MediaPipe, classifies static signs using a trained ML model, and outputs the recognized Arabic text on a Streamlit interface.

## 🎯 Key Features  
- 🔴 Real-time webcam capture (Streamlit)  
- 🟢 High-speed hand landmark extraction (MediaPipe Hands)  
- 🔵 Static sign classification using a lightweight ML model (MLP/CNN)  
- 🟣 Instant Arabic text output with confidence scoring  
- 🟡 History panel showing recent predictions  
- 🧩 Modular, scalable architecture  

## 📦 Tech Stack  
- **Python 3.10+**  
- **Streamlit**  
- **MediaPipe Hands**  
- **TensorFlow / PyTorch**  
- **OpenCV**  
- **NumPy / Pandas**  

## 📊 Dataset  
Model trained using public Arabic Sign Language datasets such as:  
- **ArASL Dataset**  
- **Arabic Sign Language Unaugmented Dataset (Kaggle)**  

Additional augmentation applied to enhance lighting, angle, and background robustness.

## 📈 Performance Targets  
- ✔️ ≥ 90% accuracy on test vocabulary  
- ✔️ ≤ 1 second inference time (CPU)  
- ✔️ Stable real-time prediction for 30+ minutes  

## 🏛 System Architecture  
**Webcam Input → MediaPipe Hands → Landmark Preprocessing → ML Classifier → Text Output → Streamlit UI**

Components include:  
- Webcam capture  
- Landmark extraction  
- Classification model  
- UI + prediction display  
- Optional admin panel for model retraining  

## 📘 Deliverables  
- Streamlit web app  
- Trained sign classification model  
- Data preprocessing + training scripts  
- Technical documentation  
- Demo video link (if included)  

## 🌱 Future Enhancements  
- Dynamic sign recognition (continuous sequences)  
- Sentence-level translation  
- Multi-language support  
- Cloud deployment (AWS/GCP)  
- Signer-independent training at scale  
