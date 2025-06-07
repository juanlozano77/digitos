# app.py
# -----------------------------------------------------------------------------
# --- 1. IMPORTS NECESARIOS ---------------------------------------------------
# -----------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, render_template
import os

# --- Desactivar logs de información de TensorFlow ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -----------------------------------------------------------------------------
# --- 2. CONFIGURACIÓN INICIAL DE FLASK Y CARGA DEL MODELO --------------------
# -----------------------------------------------------------------------------

# --- Inicializar la aplicación Flask ---
app = Flask(__name__)

# --- Cargar el modelo de Keras pre-entrenado ---
try:
    print("Cargando el modelo de Keras...")
    modelo_cargado = tf.keras.models.load_model('modelo.keras')
    print("¡Modelo cargado exitosamente!")
except Exception as e:
    print(f"Error crítico: No se pudo cargar 'modelo.keras'. Asegúrate de que el archivo esté en el directorio correcto.")
    print(f"Error: {e}")
    exit()

# -----------------------------------------------------------------------------
# --- 3. FUNCIONES DE PREPROCESAMIENTO Y UTILIDADES ---------------------------
# -----------------------------------------------------------------------------

def imagen_pil_a_base64(img):
    """Convierte una imagen de PIL a una cadena base64 para enviarla como JSON."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def preprocesar_imagen(img_data):
    """
    Decodifica una imagen en base64, la procesa y devuelve tanto el vector
    para el modelo como la imagen PIL procesada para visualización.
    """
    try:
        header, encoded = img_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(img_bytes))
        
        # --- Pasos de preprocesamiento para que coincida con MNIST ---
        img_gris = img.convert('L')
        img_redimensionada = img_gris.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Guardamos la imagen PIL procesada ANTES de convertirla en vector
        imagen_procesada_pil = img_redimensionada
        
        img_array = np.array(imagen_procesada_pil, dtype='float32')
        img_vector = img_array.flatten()
        img_normalizada = img_vector / 255.0
        img_final_vector = np.reshape(img_normalizada, (1, 784))
        
        # Devolvemos tanto el vector para el modelo como la imagen para mostrar
        return img_final_vector, imagen_procesada_pil

    except Exception as e:
        print(f"Error al preprocesar la imagen: {e}")
        return None, None

# -----------------------------------------------------------------------------
# --- 4. RUTAS DE LA APLICACIÓN (ENDPOINTS) -----------------------------------
# -----------------------------------------------------------------------------

@app.route('/')
def index():
    """Sirve la página principal de la aplicación."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Recibe la imagen, la procesa y devuelve la predicción Y la imagen procesada.
    """
    try:
        img_data = request.json['image']
        
        # Ahora obtenemos dos valores de la función de preprocesamiento
        vector_procesado, imagen_pil = preprocesar_imagen(img_data)
        
        if vector_procesado is None:
            return jsonify({'error': 'No se pudo procesar la imagen'}), 400

        # Realizar la predicción
        prediccion_raw = modelo_cargado.predict(vector_procesado)
        digito_predicho = int(np.argmax(prediccion_raw))
        
        # Convertir la imagen PIL a base64 para enviarla al frontend
        imagen_b64 = imagen_pil_a_base64(imagen_pil)
        
        # Devolver ambos resultados en el JSON
        return jsonify({
            'prediction': digito_predicho,
            'image_processed': imagen_b64
        })

    except Exception as e:
        print(f"Error en el endpoint /predict: {e}")
        return jsonify({'error': 'Ocurrió un error en el servidor'}), 500

# -----------------------------------------------------------------------------
# --- 5. PUNTO DE ENTRADA PRINCIPAL -------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)