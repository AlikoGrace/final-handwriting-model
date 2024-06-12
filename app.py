from flask import Flask, request,render_template, flash, jsonify
import os
import logging
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
# from keras.models import load_model
from tensorflow.keras.preprocessing import image

# model = load_model(r'handwriting_classification_model.h5', compile=False)
# model.compile(loss='categorical_crossentropy', compile=False)

def preprocess_image(img):
  img = img.convert('L')  # Convert to grayscale
  img = img.resize((32, 32))  # Resize to 32x32 pixels
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img = img / 255.0  # Normalize
  return img

app = Flask(__name__)
app.secret_key = "supersecretkey"


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
   
    file = request.files['file']
    if file.filename == '':
        flash("No selected file")
        return jsonify({'error': 'Error processing the image'})
    
    if file and allowed_file(file.filename):
        try:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logging.info(f"Saving file to {file_path}")
            file.save(file_path)
            model = load_model('handwriting_classification_model.h5', compile=False)
            image = Image.open(file_path)
            preprocessed_img = preprocess_image(image)
            prediction = model.predict(preprocessed_img)
            predicted_class = np.argmax(prediction, axis=1)[0]

            class_labels = {0: 'Normal', 1: 'Reversal', 2: 'Corrected'}
            result = class_labels[predicted_class]
            if result == 'Reversal' or result == 'Corrected':
              risk = 'At risk of dyslexia'
            else:
              risk = 'Not at risk of dyslexia'
            os.remove(file_path)
            return {
            "Model Prediction": risk}

            
        except Exception as e:
            logging.error(f"Error processing the image: {e}")
            return {"Error": str(e)}

    return jsonify({"error": "failed"})

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
