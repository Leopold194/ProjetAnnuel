import projetannuel
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import requests as req
from PIL import Image
import io
import pickle
import os

app = Flask(__name__)
CORS(app)


fd = open("ProjetAnnuel/dataset/images_flat_2.json","r")
data = json.load(fd)
fd.close()


labels = [x["genre"] for x in data]
pa_labels = projetannuel.string_labels(labels)

pa_x = [x["image"] for x in data]
pa_x = np.array(pa_x)

model = projetannuel.LinearModel(
    pa_x,
    pa_labels
)

print("Training model...")

model.train_classification(epochs=100, learning_rate=0.1)
print("Model trained.")


def image_to_tensor(image):
    image = image.resize((20, 20))
    image_array = np.array(image, dtype=np.float32)
    image_array /= 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    arr = np.expand_dims(image_array, axis=0)
    return arr.flatten()

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    response = req.get(url)
    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch image'}), 400
    image = response.content
    image = Image.open(io.BytesIO(image))
    image = image.convert('RGB')
    image = image_to_tensor(image)
    prediction = model.predict(image)
    return jsonify({'prediction': prediction})
    
if __name__ == '__main__':
    app.run()