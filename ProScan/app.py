from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
import os
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import bcrypt

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DB
client = MongoClient("mongodb://localhost:27017/")
db = client["proscan"]
collection = db["history"]
users_collection = db["users"]

# MODEL
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- REGISTER ----------------
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    if users_collection.find_one({"email": data["email"]}):
        return jsonify({"error": "User already exists"}), 409

    hashed = bcrypt.hashpw(data["password"].encode(), bcrypt.gensalt())

    users_collection.insert_one({
        "name": data["name"],
        "email": data["email"],
        "password": hashed.decode()
    })

    return jsonify({"message": "Registered successfully"})

# ---------------- LOGIN ----------------
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    user = users_collection.find_one({"email": data["email"]})

    if not user:
        return jsonify({"error": "User not found"}), 404

    if not bcrypt.checkpw(data["password"].encode(), user["password"].encode()):
        return jsonify({"error": "Invalid password"}), 401

    return jsonify({
        "name": user["name"],
        "email": user["email"]
    })

# ---------------- UPLOAD ----------------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]

    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        probs = F.softmax(out, dim=1)

    original = probs[0][1].item() * 100
    fake = 100 - original

    result = {
        "original": round(original, 2),
        "fake": round(fake, 2)
    }

    collection.insert_one({
        "image": filename,
        "original": result["original"],
        "fake": result["fake"],
        "date": datetime.now()
    })

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)