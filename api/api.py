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
import ijson

db.init_tables()

def compress(text: str) -> bytes:
    """Compress text using LZMA (best ratio)."""
    return lzma.compress(text.encode('utf-8'))

def uncompress(data: bytes) -> str:
    """Uncompress LZMA compressed data."""
    return lzma.decompress(data).decode('utf-8')

def get_model_name(json_string):
    f = io.StringIO(json_string)
    parser = ijson.parse(f)
    for prefix, event, value in parser:
        if prefix == 'model_name' and event == 'string':
            return value

app = Flask(__name__)
CORS(app)

def get_model_class(class_name: str):
    class_name = class_name.upper()
    if class_name == "RBF":
        return projetannuel.RBF
    elif class_name == "MLP":
        return projetannuel.MLP
    elif class_name == "LINEARMODEL":
        return projetannuel.LinearModel
    elif class_name == "SVM":
        return projetannuel.SVM
    elif class_name == "SVMOVO" :
        return projetannuel.SVMOvO
    elif class_name == "SVMOVR" :
        return projetannuel.SVMOvR


def image_to_tensor(image):
    image = image.resize((15, 10))
    image_array = np.array(image, dtype=np.float32)
    image_array /= 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    arr = np.expand_dims(image_array, axis=0)
    return arr.flatten()


class ModelWrapper:
    def __init__(self, model,name):
        self.model = model
        self.uses = 0
        self.createdon = time.time()
        self.name = name
    
    @classmethod
    def from_name(cls, name):
        weights, classname = db.get_weughts_by_model_name(model_name=name)
        if weights is None:
            raise ValueError(f"Model with name {name} not found")
        
        return cls(
            get_model_class(classname).load_string(uncompress(weights)),
            name
        )

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
        return any(m.name == model for m in self.models)
    
    def predict(self, image, model):
        self.clean()

        if not model in self :
            self.load(ModelWrapper.from_name(model))

        
        for m in self.models:
            if m.name == model:
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

@app.route("/api/model/upload", methods=["POST"])
def upload_file():    
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "Missing file"}), 400

    filename = file.filename
    name = request.form.get("name", filename)
    model_type = request.form.get("model_type")

    content = file.read()
    if filename is None or content is None:
        return jsonify({"error": "Missing file data"}), 400

    content_decoded = content.decode('utf-8') if isinstance(content, bytes) else content

    if model_type == "auto" :
        model_type = get_model_name(content_decoded)

    if model_type is None:
        return jsonify({"error": "Missing model type"}), 400


    db.upload_model(
        name=name,
        weights=compress(content_decoded),
        model_type=model_type
    )

    return jsonify({"message": "File uploaded successfully"
        , "model_type": model_type, "name": name
    }), 201

@app.route("/api/images/upload", methods=["POST"])
def upload_image():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "Missing file"}), 400

    filename = file.filename
    content = file.read()
    if filename is None or content is None:
        return jsonify({"error": "Missing file data"}), 400
    
    try :
        blob = io.BytesIO()
        image = Image.open(io.BytesIO(content))
        image = image.convert('RGB')
        image.thumbnail((100, 100), Image.Resampling.LANCZOS)
        image.save(blob, format='PNG')
        blob.seek(0)
        file_uuid = db.upload_file(blob.getvalue())
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": f"Failed to process image"}), 500


    return jsonify({"message": "File uploaded successfully", "uuid": file_uuid}), 201

@app.route("/api/document/retrieve/<uuid>", methods=["GET"])
def retrieve_document(uuid):
    content = db.retrieve_file(uuid)
    if content is None:
        return jsonify({"error": "File not found"}), 404

    return content, 200, {
                'Content-Type': 'image/png'
    }
    
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