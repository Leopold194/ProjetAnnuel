import projetannuel
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import requests as req
from PIL import Image
import io
import os
import db
import lzma
import time
from typing import List

db.init_tables()

def compress(text: str) -> bytes:
    """Compress text using LZMA (best ratio)."""
    return lzma.compress(text.encode('utf-8'))

def uncompress(data: bytes) -> str:
    """Uncompress LZMA compressed data."""
    return lzma.decompress(data).decode('utf-8')

app = Flask(__name__)
CORS(app)


def image_to_tensor(image):
    image = image.resize((20, 20))
    image_array = np.array(image, dtype=np.float32)
    image_array /= 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    arr = np.expand_dims(image_array, axis=0)
    return arr.flatten()


class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.uses = 0
        self.createdon = time.time()
    
    @classmethod
    def from_name(cls, name):
        weights = db.get_weughts_by_model_name(name)
        if weights is None:
            raise ValueError(f"Model with name {name} not found")
        return cls(projetannuel.LinearModel.load_string(uncompress(weights)))

class ModelCollection :
    def __init__(self) :
        self.models : List[ModelWrapper] = []
    
    def load(self,model: ModelWrapper):
        self.models.append(model)
    
    def clean(self,max_models: int = 10):
        self.models.sort(key=lambda x: x.uses, reverse=True)
        if len(self.models) > max_models:
            self.models = self.models[:max_models]

    def __contains__(self, model):
        return any(m.model == model for m in self.models)
    
    def predict(self, image, model):
        self.clean()

        if not model in self :
            self.load(ModelWrapper.from_name(model))
        
        for m in self.models:
            if m.model == model:
                m.uses += 1
                return m.model.predict(image)
        raise ValueError("Model not found in collection")

COLLECTION = ModelCollection()

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url')
    model = data.get('model', 'main.json')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    response = req.get(url)
    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch image'}), 400
    image = response.content
    image = Image.open(io.BytesIO(image))
    image = image.convert('RGB')
    image = image_to_tensor(image)
    prediction = COLLECTION.predict(image, model)
    return jsonify({'prediction': prediction})

@app.route("/api/upload", methods=["POST"])
def upload_file():    
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "Missing file"}), 400

    filename = file.filename
    name = request.form.get("name", filename)
    content = file.read()
    if filename is None or content is None:
        return jsonify({"error": "Missing file data"}), 400

    db.upload_model(
        name=name,
        weights=compress(content.decode('utf-8') if isinstance(content, bytes) else content)
    )

    return jsonify({"message": "File uploaded successfully"}), 201
    
@app.route("/api/models/list", methods=["GET"])
def list_models():
    models = db.list_models()
    return jsonify({"models": [
        {
            "id": model[0],
            "name": model[1],
            "created_at": model[2].isoformat()
        } for model in models
    ]}), 200

if __name__ == '__main__':
    app.run()