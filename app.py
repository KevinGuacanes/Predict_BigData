from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from flask_cors import CORS  # Para permitir solicitudes de otros dominios

# Cargar el modelo entrenado
model = load_model('model_Mnist_LeNet.h5')

# Crear la aplicación Flask
app = Flask(__name__)
CORS(app)  # Habilitar CORS para permitir solicitudes desde el frontend

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

if __name__ == '__main__':
    app.run(debug=True)
