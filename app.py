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

app = Flask(__name__)

try:
    print("Cargando el modelo de Keras...")
    modelo_cargado = tf.keras.models.load_model('modelo.keras')
    print("¡Modelo cargado exitosamente!")
except Exception as e:
    print(f"Error crítico: No se pudo cargar 'modelo.keras'.")
    print(f"Error: {e}")
    exit()

# -----------------------------------------------------------------------------
# --- 3. NUEVAS FUNCIONES DE PREPROCESAMIENTO DE IMAGEN (LAS QUE PROPORCIONASTE) ---
# -----------------------------------------------------------------------------

def center_digit(img):
    """
    Centra el dígito en una imagen 28x28 manteniendo su proporción,
    imitando el preprocesamiento de MNIST.
    """
    arr = np.array(img)

    # El dibujo viene en negro sobre transparente (o blanco).
    # Necesitamos encontrar los píxeles que no son completamente negros.
    # El canal Alfa (A) en RGBA nos dice la opacidad. Si es > 0, es parte del trazo.
    # Si la imagen no tiene alfa, usamos la luminosidad.
    if arr.shape[2] == 4: # RGBA
        coords = np.column_stack(np.where(arr[:, :, 3] > 0))
    else: # RGB o L
        # Convertimos a escala de grises para estar seguros
        img_gris = img.convert('L')
        arr_gris = np.array(img_gris)
        coords = np.column_stack(np.where(arr_gris > 0))

    if coords.size == 0:
        # Si no hay nada dibujado, devolvemos una imagen en negro
        return Image.new('L', (28, 28), color=0)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Recortar la imagen al bounding box del dígito
    # Usamos la imagen original para mantener la información de color/alfa
    crop = img.crop((x_min, y_min, x_max + 1, y_max + 1))
    
    # Convertir a escala de grises y luego invertir colores (trazo blanco sobre fondo negro)
    crop = crop.convert('L')
    
    # Redimensionar manteniendo la proporción para que quepa en un cuadro de 20x20
    h, w = crop.size
    scale = 20.0 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    crop_img = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Crear un lienzo negro de 28x28 y pegar el dígito centrado
    centered = Image.new('L', (28, 28), color=0)
    offset = ((28 - new_w) // 2, (28 - new_h) // 2)
    centered.paste(crop_img, offset)

    return centered

def preprocesar_imagen_final(img_data):
    """
    Función principal que decodifica, centra, y prepara la imagen para el modelo.
    """
    try:
        header, encoded = img_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(img_bytes))

        # 1. Usar la nueva función para centrar el dígito
        imagen_procesada_pil = center_digit(img)
        
        # 2. Convertir la imagen final (ya centrada y 28x28) a un array de NumPy
        img_array = np.array(imagen_procesada_pil, dtype='float32')
        
        # 3. Aplanar el array a un vector de 784 elementos
        img_vector = img_array.flatten()
        
        # 4. Normalizar los valores (de 0-255 a 0-1)
        img_normalizada = img_vector / 255.0
        
        # 5. Reorganizar para el formato del modelo: (1, 784)
        img_final_vector = np.reshape(img_normalizada, (1, 784))
        
        return img_final_vector, imagen_procesada_pil

    except Exception as e:
        print(f"Error al preprocesar la imagen: {e}")
        return None, None
        
def image_to_base64(img):
    """Convierte una imagen PIL a base64 para mostrarla en HTML."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# -----------------------------------------------------------------------------
# --- 4. RUTAS DE LA APLICACIÓN (ENDPOINTS) -----------------------------------
# -----------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        img_data = request.json['image']
        
        # Llamamos a la nueva función de preprocesamiento
        vector_procesado, imagen_pil = preprocesar_imagen_final(img_data)
        
        if vector_procesado is None:
            return jsonify({'error': 'No se pudo procesar la imagen'}), 400

        prediccion_raw = modelo_cargado.predict(vector_procesado)
        digito_predicho = int(np.argmax(prediccion_raw))
        
        imagen_b64 = image_to_base64(imagen_pil)
        
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
