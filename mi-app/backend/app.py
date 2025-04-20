from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from flask_cors import CORS
import os

# Cargar el modelo desde la carpeta 'models'
model = load_model('models/model_Mnist_LeNet.h5')

# Crear la aplicación Flask
app = Flask(__name__, static_folder='frontend', static_url_path='/')

CORS(app)  # Habilitar CORS para permitir solicitudes desde otros dominios

# Ruta para hacer predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener la imagen del request
    img_file = request.files['image']
    img = Image.open(io.BytesIO(img_file.read()))

    # Preprocesar la imagen
    img = img.convert('L')  # Convertir a escala de grises
    img = img.resize((28, 28))  # Redimensionar a 28x28 píxeles
    img = np.array(img) / 255.0  # Normalizar los valores
    img = img.reshape(1, 28, 28, 1)  # Cambiar la forma para que sea compatible con el modelo

    # Hacer la predicción
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Retornar el resultado
    return jsonify({'prediction': int(predicted_class)})

# Ruta para servir los archivos estáticos (HTML, CSS, JS)
@app.route('/')
def serve_frontend():
    return send_from_directory(os.path.join(app.static_folder), 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
