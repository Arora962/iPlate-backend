from flask import Flask, request, jsonify, send_from_directory
import os
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import re

# Initialize Flask app
app = Flask(__name__, static_folder='YOLO_images', static_url_path='/YOLO_images')

# Configure upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load YOLO model
model = YOLO("yolo11n.pt")  # Replace with actual YOLOv11 weights

# Load food data from the uploaded Excel file
file_path = "Anuvaad_INDB_2024.11.xlsx"  # Path to uploaded file
try:
    food_data = pd.read_excel(file_path, sheet_name='Sheet1')
    if 'food_name' in food_data.columns:
        food_data['food_name'] = food_data['food_name'].astype(str).str.lower()
        food_data['normalized_food_name'] = food_data['food_name'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    else:
        raise KeyError("Column 'food_name' not found in Excel file.")
except FileNotFoundError:
    print(f"[ERROR] Excel file {file_path} not found.")
    food_data = pd.DataFrame()
except Exception as e:
    print(f"[ERROR] Failed to load Excel file: {str(e)}")
    food_data = pd.DataFrame()

def normalize_text(text):
    """Normalize text by converting to lowercase and removing special characters."""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower().strip())

def get_food_details(food_name):
    """Fetch food details by matching the detected object's name in the food_name column."""
    if food_data.empty:
        print("[ERROR] Food dataset is empty.")
        return None

    if 'food_name' not in food_data.columns:
        print("[ERROR] Column 'food_name' not found in the dataset.")
        return None

    normalized_food_name = normalize_text(food_name)
    match = food_data[food_data['normalized_food_name'] == normalized_food_name]

    if match.empty:
        match = food_data[food_data['normalized_food_name'].str.contains(normalized_food_name, na=False)]

    if not match.empty:
        print(f"[INFO] Found Match for '{food_name}': {match.iloc[0]['food_name']}")
        return match.iloc[0].to_dict()
    else:
        print(f"[WARNING] No match found for '{food_name}' in dataset.")
    return None

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Flask YOLOv11 Food Detection API.'})

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '' or not allowed_file(image_file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    filename = secure_filename(image_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        # Save the uploaded image
        image_file.save(filepath)
        print(f"[INFO] Image saved: {filepath}")
        
        # Call the /detect endpoint after the image is saved
        return detect_food(filename)  # Call the detection function with the saved image filename

    except Exception as e:
        return jsonify({'error': f'File save failed: {str(e)}'}), 500

@app.route('/detect/<filename>', methods=['GET'])
def detect_food(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        print(f"[ERROR] Image not found: {filepath}")
        return jsonify({'error': 'Image not found'}), 404

    try:
        image = Image.open(filepath)  # Open the image using PIL
        results = model(image)  # Run the YOLO detection on the image

        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id].strip()
                print(f"[INFO] Detected Class: {class_name}")
                food_details = get_food_details(class_name)

                detection_info = {
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),
                    "food_details": food_details if food_details else "Not found in database"
                }
                detections.append(detection_info)

        return jsonify({
            'message': 'Object detection completed',
            'detections': detections
        }), 200

    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500



@app.route('/uploads/<filename>', methods=['GET'])
def get_uploaded_image(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        print(f"[ERROR] Requested file not found: {filepath}")
        return jsonify({'error': f'File {filename} not found'}), 404

    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
