from flask import Flask, request, jsonify
import os
import cv2
import numpy as np

app = Flask(__name__)

dataset_path = "dataset/"
trainer_path = "trainer.yml"

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ensure dataset directory exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

@app.route('/upload_faces', methods=['POST'])
def upload_faces():
    """Receive multiple images for a person and store them."""
    person_name = request.form.get('name')
    if not person_name:
        return jsonify({"error": "Name is required"}), 400
    
    person_dir = os.path.join(dataset_path, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images received"}), 400
    
    for file in files:
        img_path = os.path.join(person_dir, file.filename)
        file.save(img_path)

    return jsonify({"message": "Images saved for " + person_name}), 200

@app.route('/train', methods=['GET'])
def train_model():
    """Train the LBPH recognizer with stored images."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    label_map = {}
    label_count = 0

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (200, 200))  # Ensure consistent image size
            faces.append(img)
            if person_name not in label_map:
                label_map[person_name] = label_count
                label_count += 1
            labels.append(label_map[person_name])

    recognizer.train(faces, np.array(labels))
    recognizer.save(trainer_path)
    
    return jsonify({"message": "Training complete", "people_trained": list(label_map.keys())}), 200

@app.route('/detect', methods=['POST'])
def detect_face():
    """Detect a face in an uploaded image and return the closest trained match."""
    if not os.path.exists(trainer_path):
        return jsonify({"error": "Model not trained yet"}), 400

    file = request.files['image']
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path)

    faces = face_detector.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return jsonify({"message": "No face detected"}), 400

    # Correct label mapping
    label_map = {}
    for idx, person_name in enumerate(os.listdir(dataset_path)):
        label_map[idx] = person_name

    best_match = None
    best_confidence = float("inf")

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))  # Ensure consistent size
        label, confidence = recognizer.predict(face)

        if confidence < best_confidence:
            best_confidence = confidence
            best_match = label_map.get(label, "Unknown")  # Always returns a trained person

    return jsonify({"recognized": best_match, "confidence": best_confidence}), 200
if __name__ == '__main__':
    print("Starting Flask server...")  # Debug message
    app.run(host='0.0.0.0', port=5000, debug=True)
