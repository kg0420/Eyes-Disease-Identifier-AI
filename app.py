import os, uuid
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime

# ---------------- CONFIG ----------------
ALLOWED_EXT = {"jpg", "jpeg", "png"}
UPLOAD_DIR = os.path.join("static", "uploads")
GRADCAM_DIR = os.path.join("static", "gradcam")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

IMG_SIZE = (256, 256)

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "eyes_disease_detection.h5"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

# ---------------- HELPERS ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Unable to read image")

    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- GRAD-CAM ----------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_gradcam(image_path, heatmap, alpha=0.4):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)

    filename = f"{uuid.uuid4().hex}_gradcam.jpg"
    save_path = os.path.join(GRADCAM_DIR, filename)
    cv2.imwrite(save_path, superimposed)

    return f"/static/gradcam/{filename}"

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    f = request.files["image"]

    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(f.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(f.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    f.save(save_path)

    try:
        img_array = preprocess_image(save_path)
        preds = model.predict(img_array)[0]

        class_id = int(np.argmax(preds))
        label = CLASS_NAMES[class_id]
        conf = float(preds[class_id] * 100)

        probs = {
            CLASS_NAMES[i]: float(preds[i] * 100)
            for i in range(len(CLASS_NAMES))
        }

        # -------- Grad-CAM --------
        last_conv_layer_name = "conv2d_9"  # ⚠️ I will auto-detect this for you
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        gradcam_url = save_gradcam(save_path, heatmap)

        return jsonify({
            "label": label,
            "confidence": round(conf, 2),
            "probs": probs,
            "image_url": f"/static/uploads/{unique_name}",
            "gradcam_url": gradcam_url,
            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")
        })

    except Exception as e:
        print("Inference error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5002)
