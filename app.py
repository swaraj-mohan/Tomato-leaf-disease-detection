from __future__ import division, print_function
import os
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model_InceptionV3.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)

    # Scaling
    x = x/255
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "The plant is infected with Bacterial spot"
    elif preds == 1:
        preds = "The plant is infected with Early blight"
    elif preds == 2:
        preds = "The plant is healthy"
    elif preds == 3:
        preds = "The plant is infected with Late blight"
    elif preds == 4:
        preds = "The plant is infected with Leaf Mold"
    elif preds == 5:
        preds = "The plant is infected with Septoria leaf spot"
    elif preds == 6:
        preds = "The plant is infected with Spider mites Two-spotted spider mite"
    elif preds == 7:
        preds = "The plant is infected with Target Spot"
    elif preds == 8:
        preds = "The plant is infected with Tomato mosaic virus"
    else:
        preds = "The plant is infected with Tomato Yellow Leaf Curl Virus"
        
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)