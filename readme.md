# 👁️ Eye Disease Detection System – Doctor Dashboard

An AI-powered web application for detecting eye diseases from retinal/eye images using deep learning. This system provides a **hospital-style doctor dashboard**, real-time predictions, disease probability charts, and an intuitive user interface.

---

## 🚀 Features

- 📊 Disease Probability Visualization (Chart.js)
- 🏥 Doctor Dashboard UI (Hospital-grade design)
- 📷 Image Upload Support
- 📈 Class-wise Confidence Scores
- 🌙 Dark Themed Professional Interface
- 📱 Fully Responsive (Mobile, Tablet, Desktop)
- 🔐 Secure File Uploads
- ⚡ Fast Inference using ONNX Runtime

---

## 🩺 Supported Classes

- Cataract
- Diabetic Retinopathy
- Glaucoma
- Normal

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| Backend  | Flask |
| Model Inference | ONNX Runtime |
| Frontend | HTML, TailwindCSS |
| Charts | Chart.js |
| Image Processing | OpenCV |
| Language | Python |

---

## 📁 Project Structure

eye-disease-detection/
│
├── static/
│ └── uploads/
│
├── templates/
│ └── index.html
│
├── model.onnx
├── app.py
├── requirements.txt
└── README.md

yaml
Copy code

---

## ⚙️ Installation

### 1️⃣ Clone the Repository


git clone https://github.com/kg0420/Eyes-Disease-Identifier-AI
cd eye-disease-detection
2️⃣ Create Virtual Environment (Optional but Recommended)
bash
Copy code
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the Application
bash
Copy code
python app.py
5️⃣ Open in Browser
cpp
Copy code
http://127.0.0.1:5000/
🧪 How It Works
Upload an eye/retinal image

Image is preprocessed

ONNX model performs inference

Class probabilities are computed

Result + confidence displayed

Bar chart visualizes all probabilities

📸 UI Preview
Doctor Dashboard

Prediction Panel

Disease Probability Chart

Disease Info Cards

Dark Medical Theme

🔐 Security
Secure file uploads

Allowed file types: JPG, JPEG, PNG

Randomized filenames

Server-side validation

📦 requirements.txt (Example)
nginx
Copy code
flask
opencv-python
numpy
onnxruntime
🧑‍⚕️ Future Enhancements
🔥 Grad-CAM Heatmap Visualization

📄 Medical PDF Report Download

🗂 Patient History

🧑‍⚕️ Doctor Login System

🌍 Multi-language Support

🎙 Voice Output

📱 Mobile App Version

👨‍💻 Developed By
Krish Gupta
AI & ML Developer

📜 License
This project is licensed under the MIT License.

⭐ Support
If you like this project, please ⭐ star the repository and share it!

❓ Need Help?
Feel free to contact me or open an issue!



