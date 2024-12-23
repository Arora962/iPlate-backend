from flask import Flask, request, jsonify
import os
from PIL import Image
import io

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Flask Image Processing API!'})

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    try:
        image = Image.open(image_file)

        processed_image = image.convert("L")

        buffer = io.BytesIO()
        processed_image.save(buffer, format="JPEG")
        buffer.seek(0)

        return jsonify({'message': 'Image processed successfully!'}), 200

    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
