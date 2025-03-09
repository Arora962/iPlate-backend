# from flask import Flask, request, jsonify
# import os
# from PIL import Image
# import io

# app = Flask(__name__)

# @app.route('/', methods=['GET'])
# def home():
#     return jsonify({'message': 'Welcome to the Flask Image Processing API!'})

# @app.route('/process-image', methods=['POST'])
# def process_image():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file provided'}), 400

#     image_file = request.files['image']

#     try:
#         image = Image.open(image_file)

#         processed_image = image.convert("L")

#         buffer = io.BytesIO()
#         processed_image.save(buffer, format="JPEG")
#         buffer.seek(0)

#         return jsonify({'message': 'Image processed successfully!'}), 200

#     except Exception as e:
#         return jsonify({'error': f'Failed to process image: {str(e)}'}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)



# from flask import Flask, request, jsonify, send_from_directory
# import os
# import torch
# from PIL import Image
# import io
# from ultralytics import YOLO
# from werkzeug.utils import secure_filename

# # Initialize Flask app
# app = Flask(__name__)

# # Configure upload folder
# UPLOAD_FOLDER = "uploads"
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# # Load YOLOv8/YOLO11n model
# model = YOLO("yolo11n.pt") 

# @app.route('/', methods=['GET'])
# def home():
#     return jsonify({'message': 'Welcome to the Flask YOLOv8/YOLO11n.'})

# # Upload image and save to server
# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file provided'}), 400

#     image_file = request.files['image']
#     filename = secure_filename(image_file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

#     try:
#         image_file.save(filepath)
#         return jsonify({'message': 'Image uploaded successfully!', 'file_path': filepath}), 200
#     except Exception as e:
#         return jsonify({'error': f'Failed to upload image: {str(e)}'}), 500

# # Retrieve and scan uploaded image
# @app.route('/scan/<filename>', methods=['GET'])
# def scan_uploaded_image(filename):
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
#     if not os.path.exists(filepath):
#         return jsonify({'error': 'Image not found'}), 404

#     try:
#         # Open and process image with YOLOv8/YOLO11n
#         image = Image.open(filepath)
#         results = model(image)

#         detections = []
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
#                 confidence = float(box.conf[0])  # Confidence score
#                 class_id = int(box.cls[0])  # Class ID
#                 class_name = model.names[class_id]  # Class name

#                 detections.append({
#                     "class_name": class_name,
#                     "class_id": class_id,
#                     "confidence": confidence,
#                     "bbox": [x1, y1, x2, y2]
#                 })

#         return jsonify({'message': 'Object detection completed', 'detections': detections}), 200

#     except Exception as e:
#         return jsonify({'error': f'Failed to scan image: {str(e)}'}), 500

# # Detect objects directly from uploaded image in request
# @app.route('/detect', methods=['POST'])
# def detect_objects():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file provided'}), 400

#     image_file = request.files['image']

#     try:
#         # Open image
#         image = Image.open(image_file)

#         # Run YOLOv8/YOLO11n object detection
#         results = model(image)

#         # Parse detection results
#         detections = []
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
#                 confidence = float(box.conf[0])  # Confidence score
#                 class_id = int(box.cls[0])  # Class ID
#                 class_name = model.names[class_id]  # Class name

#                 detections.append({
#                     "class_name": class_name,
#                     "class_id": class_id,
#                     "confidence": confidence,
#                     "bbox": [x1, y1, x2, y2]
#                 })

#         return jsonify({'message': 'Object detection completed', 'detections': detections}), 200

#     except Exception as e:
#         return jsonify({'error': f'Failed to process image: {str(e)}'}), 500

# # Serve uploaded images (optional)
# @app.route('/uploads/<filename>', methods=['GET'])
# def get_uploaded_image(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001, debug=True)