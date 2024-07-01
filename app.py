from flask import Flask, request, render_template, flash, jsonify, session
import os
import logging
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((32, 32))  # Resize to 32x32 pixels
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize
    return img

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'predictions' not in session:
        session['predictions'] = []
    
    file = request.files['file']
    if file.filename == '':
        flash("No selected file")
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logging.info(f"Saving file to {file_path}")
            file.save(file_path)
            
            model = load_model('handwriting_classification_model.h5', compile=False)
            img = Image.open(file_path)
            preprocessed_img = preprocess_image(img)
            prediction = model.predict(preprocessed_img)
            predicted_class = np.argmax(prediction, axis=1)[0]

            class_labels = {0: 'Normal', 1: 'Reversal', 2: 'Corrected'}
            result = class_labels[predicted_class]
            
            session['predictions'].append(result)
            session.modified = True  # Ensure session is saved
            logging.info(f"Current predictions: {session['predictions']}")
            os.remove(file_path)

            if len(session['predictions']) == 5:
                # Determine the final prediction
                risk_counts = {'Normal': 0, 'Reversal': 0, 'Corrected': 0}
                for pred in session['predictions']:
                    risk_counts[pred] += 1

                if risk_counts['Reversal'] > 0 or risk_counts['Corrected'] > 0:
                    risk = 'At risk of dyslexia'
                else:
                    risk = 'Not at risk of dyslexia'

                session.pop('predictions', None)  # Clear the session
                return jsonify({"Model_Prediction": risk})
            else:
                return jsonify({"message": "Image processed, please upload more images", "uploaded_images": len(session['predictions'])})
            
        except Exception as e:
            logging.error(f"Error processing the image: {e}")
            return jsonify({"Error": str(e)})
    
    return jsonify({"error": "Failed to process the image"})

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
