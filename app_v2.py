from tensorflow import keras
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
from werkzeug.utils import secure_filename

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Inicializar la app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Crear carpeta de subida si no existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Función para loguear el uso de memoria
def log_memory_usage(context=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.debug(f"[{context}] Memoria usada: {mem_mb:.2f} MB")

# Cargar el modelo
logger.info("Cargando modelo...")
log_memory_usage("Antes de cargar el modelo")
model = keras.models.load_model("model_overfit.h5")
logger.info("Modelo cargado")
log_memory_usage("Después de cargar el modelo")

# Lista de clases
names = ['ballena beluga', 'ballena azul', 'delfín mular', 'ballena bryde',
         'delfín commerson', 'delfín común', 'ballena de cuvier',
         'delfín moteado', 'falsa orca', 'ballena rorcual común', 'delfín de frasier',
         'ballena gris', 'ballena jorobada', 'orca', 'ballena cabezona', 'ballena minke',
         'delfín tropical', 'calderón', 'orca pigmea', 'delfín de dientes rugosos',
         'ballena sei', 'ballena jorobada del sur', 'delfín girador', 'delfín moteado del atlántico',
         'delfín de lados blancos del pacífico']

def model_predict(image_path, model):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    x = np.array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/predict", methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath= os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        #Predicción
        preds = model_predict(file_path, model)

        print('PREDICCIÓN', names[np.argmax(preds)])

        result = str(names[np.argmax(preds)])
        return result
    return render_template('index.html')



if __name__ == "__main__":
    app.run()
