import os
import cv2
import numpy as np
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Path settings
DATASET_PATH = "dataset"  # Folder where images are stored
MODEL_PATH = "face_recognizer.yml"  # Model save path
LABEL_DICT_PATH = "label_dict.pkl"  # Label dictionary save path

# Initialize Face Recognizer (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ensure dataset folder exists
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# Load label dictionary if it exists
if os.path.exists(LABEL_DICT_PATH):
    with open(LABEL_DICT_PATH, "rb") as f:
        label_dict = pickle.load(f)
else:
    label_dict = {}  # Initialize an empty dictionary

# -------------------- 1. Register User (Save Face Images) --------------------
@app.route("/register", methods=["POST"])
def register():
    try:
        name = request.form["name"]  # User's name
        image = request.files["image"]  # Uploaded image

        if not name or not image:
            return jsonify({"error": "Missing name or image"}), 400

        user_folder = os.path.join(DATASET_PATH, name)
        os.makedirs(user_folder, exist_ok=True)

        img_path = os.path.join(user_folder, f"{len(os.listdir(user_folder)) + 1}.jpg")
        image.save(img_path)

        return jsonify({"message": f"Image saved as {img_path}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- 2. Train Model --------------------
def get_training_data():
    faces = []
    labels = []
    label_dict = {}
    label_id = 0

    if not os.path.exists(DATASET_PATH):
        return faces, labels, label_dict  # Return empty lists safely

    for person_name in os.listdir(DATASET_PATH):
        person_folder = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_folder):
            continue

        if person_name not in label_dict:
            label_dict[person_name] = label_id
            label_id += 1

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Skipping invalid image: {image_path}")
                continue

            faces.append(img)
            labels.append(label_dict[person_name])

    return faces, labels, label_dict

@app.route("/train", methods=["POST"])
def train():
    faces, labels, label_dict = get_training_data()

    if len(faces) == 0:
        return jsonify({"error": "No faces found to train"}), 400

    recognizer.train([np.array(face, dtype=np.uint8) for face in faces], np.array(labels))
    recognizer.save(MODEL_PATH)

    # Save label dictionary
    with open(LABEL_DICT_PATH, "wb") as f:
        pickle.dump(label_dict, f)

    return jsonify({"message": "Training completed", "labels": label_dict}), 200

# -------------------- 3. Recognize Face --------------------
@app.route("/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    nparr = np.frombuffer(image.read(), np.uint8)
    img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    if img_gray is None:
        return jsonify({"error": "Invalid image"}), 400

    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained"}), 500

    # Load the label dictionary
    if not os.path.exists(LABEL_DICT_PATH):
        return jsonify({"error": "Label dictionary not found"}), 500

    with open(LABEL_DICT_PATH, "rb") as f:
        label_dict = pickle.load(f)

    # Create reverse mapping (label to name)
    name_dict = {v: k for k, v in label_dict.items()}

    recognizer.read(MODEL_PATH)

    faces_detected = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    if len(faces_detected) == 0:
        return jsonify({"message": "No face detected"}), 404

    results = []
    for (x, y, w, h) in faces_detected:
        face_roi = img_gray[y:y+h, x:x+w]

        try:
            label, confidence = recognizer.predict(face_roi)
            # Convert label to name
            name = name_dict.get(label, "Unknown")
            results.append({"name": name, "confidence": confidence})
        except Exception as e:
            print(f"Recognition error: {e}")
            results.append({"name": "Error", "confidence": None})

    return jsonify({"recognized_faces": results}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render-assigned port
    app.run(host="0.0.0.0", port=port)
